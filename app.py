# render_app.py - Optimized for Render deployment
import os
import gc
import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from flask import Flask, request, jsonify
import tempfile
import json
from werkzeug.utils import secure_filename
import logging
import psutil
from threading import Lock
import time
from functools import wraps
import sys

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB

# Configure logging for Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global encoder instance with thread safety
encoder_lock = Lock()
encoder_instance = None

def get_encoder():
    """Get or create encoder instance with thread safety"""
    global encoder_instance
    with encoder_lock:
        if encoder_instance is None:
            logger.info("Initializing voice encoder...")
            try:
                encoder_instance = VoiceEncoder()
                logger.info("Voice encoder initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize encoder: {e}")
                raise
        return encoder_instance

def memory_monitor(func):
    """Decorator to monitor memory usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except:
            initial_memory = 0
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
        finally:
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            try:
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(f"{func.__name__} - Memory: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
                
                # Warning if memory usage is high
                if final_memory > 800:  # 800MB warning for Render
                    logger.warning(f"High memory usage: {final_memory:.1f}MB")
            except:
                pass
    
    return wrapper

class OptimizedVoiceProcessor:
    def __init__(self):
        self.min_audio_length = 1.0
        self.max_audio_length = 8.0  # Reduced for Render limits
        self.verification_threshold = float(os.getenv('DEFAULT_VERIFICATION_THRESHOLD', 0.75))
        self.sample_rate = 16000
        self.max_embeddings_cache = 50  # Reduced cache size
        
    @memory_monitor
    def extract_embedding(self, audio_path):
        """Extract voice embedding with memory optimization"""
        try:
            encoder = get_encoder()
            
            # Load and validate audio
            duration = librosa.get_duration(filename=audio_path)
            
            # Validate duration
            if duration < self.min_audio_length:
                raise ValueError(f"Audio too short. Minimum {self.min_audio_length} seconds required.")
            
            if duration > self.max_audio_length:
                raise ValueError(f"Audio too long. Maximum {self.max_audio_length} seconds allowed.")
            
            logger.info(f"Processing audio: {duration:.2f}s")
            
            # Preprocess with memory-efficient approach
            wav = preprocess_wav(audio_path)
            
            # Check processed length
            if len(wav) < self.min_audio_length * self.sample_rate:
                raise ValueError("Audio too short after preprocessing.")
            
            # Extract embedding
            with encoder_lock:
                embedding = encoder.embed_utterance(wav)
            
            # Convert to list and clean up
            embedding_list = embedding.tolist()
            del wav, embedding  # Explicit cleanup
            gc.collect()
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            gc.collect()
            raise
    
    @memory_monitor 
    def compare_embeddings(self, embedding1, embedding2):
        """Compare embeddings with memory optimization"""
        try:
            # Validate inputs
            if not isinstance(embedding1, list) or not isinstance(embedding2, list):
                raise ValueError("Embeddings must be lists")
            
            if len(embedding1) != len(embedding2):
                raise ValueError("Embeddings must have same length")
            
            # Convert to numpy arrays
            emb1 = np.array(embedding1, dtype=np.float32)
            emb2 = np.array(embedding2, dtype=np.float32)
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)
            
            # Cleanup
            del emb1, emb2
            gc.collect()
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error comparing embeddings: {str(e)}")
            gc.collect()
            raise
    
    @memory_monitor
    def verify_voice(self, test_embedding, enrolled_embeddings, threshold=None):
        """Verify voice with batch processing optimization"""
        if threshold is None:
            threshold = self.verification_threshold
            
        try:
            if not enrolled_embeddings:
                raise ValueError("No enrolled embeddings provided")
            
            # Limit number of comparisons for Render
            max_comparisons = 15
            if len(enrolled_embeddings) > max_comparisons:
                logger.warning(f"Too many enrollments ({len(enrolled_embeddings)}), using first {max_comparisons}")
                enrolled_embeddings = enrolled_embeddings[:max_comparisons]
            
            similarities = []
            
            # Process in smaller batches
            batch_size = 3
            for i in range(0, len(enrolled_embeddings), batch_size):
                batch = enrolled_embeddings[i:i+batch_size]
                
                for enrolled_embedding in batch:
                    similarity = self.compare_embeddings(test_embedding, enrolled_embedding)
                    similarities.append(similarity)
                
                # Force garbage collection after each batch
                gc.collect()
            
            # Calculate statistics
            avg_similarity = np.mean(similarities) if similarities else 0.0
            max_similarity = max(similarities) if similarities else 0.0
            is_match = avg_similarity >= threshold
            
            result = {
                'is_match': is_match,
                'confidence': float(avg_similarity),
                'max_confidence': float(max_similarity),
                'threshold': threshold,
                'num_comparisons': len(similarities)
            }
            
            # Cleanup
            del similarities
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in voice verification: {str(e)}")
            gc.collect()
            raise

# Initialize processor
voice_processor = OptimizedVoiceProcessor()

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'service': 'Voice Processing API',
        'status': 'running',
        'version': '1.0.0'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check for Render"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return jsonify({
            'status': 'healthy',
            'service': 'voice_processor_render',
            'memory_mb': round(memory_info.rss / 1024 / 1024, 1),
            'encoder_loaded': encoder_instance is not None,
            'environment': os.getenv('RENDER_SERVICE_NAME', 'local')
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/extract_embedding', methods=['POST'])
@memory_monitor
def extract_embedding():
    """Memory-optimized embedding extraction for Render"""
    temp_file_path = None
    
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file size before processing
        audio_file.seek(0, 2)
        file_size = audio_file.tell()
        audio_file.seek(0)
        
        max_size = 8 * 1024 * 1024  # 8MB
        if file_size > max_size:
            return jsonify({'error': f'File too large. Maximum size is {max_size/1024/1024}MB'}), 400
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file_path = temp_file.name
            audio_file.save(temp_file_path)
        
        try:
            # Extract embedding
            embedding = voice_processor.extract_embedding(temp_file_path)
            
            return jsonify({
                'success': True,
                'embedding': embedding,
                'embedding_size': len(embedding)
            })
            
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
            
    except Exception as e:
        logger.error(f"Error in extract_embedding: {str(e)}")
        return jsonify({'error': 'Failed to process audio'}), 500
    
    finally:
        # Always cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
        
        # Force garbage collection
        gc.collect()

@app.route('/compare_embeddings', methods=['POST'])
@memory_monitor
def compare_embeddings():
    """Memory-optimized embedding comparison"""
    try:
        data = request.get_json()
        
        if not data or 'embedding1' not in data or 'embedding2' not in data:
            return jsonify({'error': 'Both embeddings required'}), 400
        
        embedding1 = data['embedding1']
        embedding2 = data['embedding2']
        
        similarity = voice_processor.compare_embeddings(embedding1, embedding2)
        
        return jsonify({
            'success': True,
            'similarity': similarity,
            'is_match': similarity >= voice_processor.verification_threshold
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in compare_embeddings: {str(e)}")
        return jsonify({'error': 'Failed to compare embeddings'}), 500

@app.route('/verify_voice', methods=['POST'])
@memory_monitor
def verify_voice():
    """Memory-optimized voice verification"""
    try:
        data = request.get_json()
        
        if not data or 'test_embedding' not in data or 'enrolled_embeddings' not in data:
            return jsonify({'error': 'Test embedding and enrolled embeddings required'}), 400
        
        test_embedding = data['test_embedding']
        enrolled_embeddings = data['enrolled_embeddings']
        threshold = data.get('threshold', voice_processor.verification_threshold)
        
        if not enrolled_embeddings:
            return jsonify({'error': 'No enrolled embeddings provided'}), 400
        
        result = voice_processor.verify_voice(test_embedding, enrolled_embeddings, threshold)
        
        return jsonify({
            'success': True,
            **result
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in verify_voice: {str(e)}")
        return jsonify({'error': 'Failed to verify voice'}), 500

@app.route('/memory_status', methods=['GET'])
def memory_status():
    """Get memory status for monitoring"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return jsonify({
            'rss_mb': round(memory_info.rss / 1024 / 1024, 1),
            'vms_mb': round(memory_info.vms / 1024 / 1024, 1),
            'encoder_loaded': encoder_instance is not None,
            'service': os.getenv('RENDER_SERVICE_NAME', 'local')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 8MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

# Memory cleanup middleware
@app.after_request
def after_request(response):
    # Light garbage collection after requests
    if gc.get_count()[0] > 50:
        gc.collect()
    return response

# Render deployment configuration
if __name__ == '__main__':
    # Environment setup for Render
    port = int(os.environ.get('PORT', 5001))
    
    # Log startup info
    logger.info(f"Starting voice processor on port {port}")
    logger.info(f"Environment: {os.getenv('RENDER_SERVICE_NAME', 'local')}")
    
    try:
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"Initial memory usage: {initial_memory:.1f}MB")
    except:
        pass
    
    # Run with Render-optimized settings
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=False,
        threaded=True
    )