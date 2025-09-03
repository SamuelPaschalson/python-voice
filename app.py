# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import librosa
# import warnings
# import tempfile
# import os
# import base64
# import soundfile as sf

# warnings.filterwarnings("ignore")

# app = Flask(__name__)
# CORS(app)

# # Fixed create_embedding function
# def create_embedding(audio_data, prompt_text=None):
#     try:
#         # Create a temporary file to save the audio
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
#             tmp_file.write(audio_data)
#             tmp_filename = tmp_file.name
        
#         # Load audio file using librosa
#         y, sr = librosa.load(tmp_filename, sr=16000)  # Resample to 16kHz
        
#         # Clean up temporary file
#         os.unlink(tmp_filename)
        
#         # Extract audio features (MFCCs as a simple example)
#         # In a real implementation, you'd use a proper voice embedding model
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
#         # Average across time to get a fixed-length vector
#         embedding = np.mean(mfccs, axis=1)
        
#         # Normalize the embedding
#         embedding = embedding / np.linalg.norm(embedding)
        
#         return {"embedding": embedding.tolist(), "status": "success"}
    
#     except Exception as e:
#         return {"error": f"Embedding creation failed: {str(e)}"}

# def cosine_similarity(vec1, vec2):
#     try:
#         vec1 = np.array(vec1)
#         vec2 = np.array(vec2)
#         dot_product = np.dot(vec1, vec2)
#         norm_vec1 = np.linalg.norm(vec1)
#         norm_vec2 = np.linalg.norm(vec2)
#         return dot_product / (norm_vec1 * norm_vec2)
#     except Exception as e:
#         return 0.0

# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({"status": "OK", "service": "voice-embedding"})

# @app.route('/create-embedding', methods=['POST'])
# def handle_create_embedding():
#     try:
#         if 'audio' not in request.files:
#             return jsonify({"error": "No audio file provided"}), 400
        
#         audio_file = request.files['audio']
#         audio_data = audio_file.read()
        
#         result = create_embedding(audio_data)
        
#         if "error" in result:
#             return jsonify(result), 400
        
#         return jsonify(result)
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Fixed text-dependent verification endpoint
# @app.route('/create-text-dependent-embedding', methods=['POST'])
# def handle_create_text_dependent_embedding():
#     try:
#         if 'audio' not in request.files:
#             return jsonify({"error": "No audio file provided"}), 400
        
#         if 'prompt_text' not in request.form:
#             return jsonify({"error": "No prompt text provided"}), 400
        
#         audio_file = request.files['audio']
#         prompt_text = request.form['prompt_text']
#         audio_data = audio_file.read()
        
#         # Call the actual embedding function
#         result = create_embedding(audio_data, prompt_text)
        
#         if "error" in result:
#             return jsonify(result), 400
        
#         return jsonify({
#             "embedding": result["embedding"],
#             "prompt_text": prompt_text,
#             "status": "success"
#         })
    
#     except Exception as e:
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# # Fixed Base64 endpoint for serverless compatibility
# @app.route('/create-text-dependent-embedding-base64', methods=['POST'])
# def handle_create_text_dependent_embedding_base64():
#     try:
#         data = request.get_json()
        
#         if not data or 'audio_base64' not in data:
#             return jsonify({"error": "No base64 audio data provided"}), 400
        
#         if 'prompt_text' not in data:
#             return jsonify({"error": "No prompt text provided"}), 400
        
#         audio_base64 = data['audio_base64']
#         prompt_text = data['prompt_text']
        
#         # Decode base64 audio
#         audio_data = base64.b64decode(audio_base64)
        
#         # Call the actual embedding function
#         result = create_embedding(audio_data, prompt_text)
        
#         if "error" in result:
#             return jsonify(result), 400
        
#         return jsonify({
#             "embedding": result["embedding"],
#             "prompt_text": prompt_text,
#             "status": "success"
#         })
    
#     except Exception as e:
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# @app.route('/compare-embeddings', methods=['POST'])
# def handle_compare_embeddings():
#     try:
#         data = request.get_json()
        
#         if not data or 'embedding1' not in data or 'embedding2' not in data:
#             return jsonify({"error": "Both embeddings are required"}), 400
        
#         embedding1 = data['embedding1']
#         embedding2 = data['embedding2']
        
#         similarity = cosine_similarity(embedding1, embedding2)
        
#         return jsonify({
#             "similarity": float(similarity),
#             "status": "success"
#         })
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

import os
import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from flask import Flask, request, jsonify
import tempfile
import json
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize voice encoder
encoder = VoiceEncoder()

class VoiceProcessor:
    def __init__(self):
        self.encoder = VoiceEncoder()
        self.min_audio_length = 1.0  # Minimum audio length in seconds
        self.verification_threshold = 0.75  # Similarity threshold for verification
    
    def extract_embedding(self, audio_path):
        """Extract voice embedding from audio file"""
        try:
            # Load and preprocess the audio
            wav = preprocess_wav(audio_path)
            
            # Check minimum audio length
            if len(wav) < self.min_audio_length * 16000:  # 16kHz sample rate
                raise ValueError("Audio too short. Minimum 1 second required.")
            
            # Extract embedding
            embedding = self.encoder.embed_utterance(wav)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            raise
    
    def compare_embeddings(self, embedding1, embedding2):
        """Compare two voice embeddings and return similarity score"""
        try:
            # Convert lists back to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error comparing embeddings: {str(e)}")
            raise
    
    def verify_voice(self, test_embedding, enrolled_embeddings, threshold=None):
        """Verify voice against enrolled embeddings"""
        if threshold is None:
            threshold = self.verification_threshold
            
        try:
            similarities = []
            for enrolled_embedding in enrolled_embeddings:
                similarity = self.compare_embeddings(test_embedding, enrolled_embedding)
                similarities.append(similarity)
            
            # Use average similarity for verification
            avg_similarity = np.mean(similarities) if similarities else 0.0
            max_similarity = max(similarities) if similarities else 0.0
            
            is_match = avg_similarity >= threshold
            
            return {
                'is_match': is_match,
                'confidence': float(avg_similarity),
                'max_confidence': float(max_similarity),
                'threshold': threshold,
                'num_comparisons': len(similarities)
            }
            
        except Exception as e:
            logger.error(f"Error in voice verification: {str(e)}")
            raise

# Initialize voice processor
voice_processor = VoiceProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'voice_processor'})

@app.route('/extract_embedding', methods=['POST'])
def extract_embedding():
    """Extract voice embedding from uploaded audio file"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            
            try:
                # Extract embedding
                embedding = voice_processor.extract_embedding(temp_file.name)
                
                return jsonify({
                    'success': True,
                    'embedding': embedding,
                    'embedding_size': len(embedding)
                })
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)
                
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in extract_embedding: {str(e)}")
        return jsonify({'error': 'Failed to process audio'}), 500

@app.route('/compare_embeddings', methods=['POST'])
def compare_embeddings():
    """Compare two voice embeddings"""
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
        
    except Exception as e:
        logger.error(f"Error in compare_embeddings: {str(e)}")
        return jsonify({'error': 'Failed to compare embeddings'}), 500

@app.route('/verify_voice', methods=['POST'])
def verify_voice():
    """Verify voice against multiple enrolled embeddings"""
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
        
    except Exception as e:
        logger.error(f"Error in verify_voice: {str(e)}")
        return jsonify({'error': 'Failed to verify voice'}), 500

@app.route('/batch_extract', methods=['POST'])
def batch_extract_embeddings():
    """Extract embeddings from multiple audio files"""
    try:
        files = request.files.getlist('audio_files')
        
        if not files:
            return jsonify({'error': 'No audio files provided'}), 400
        
        embeddings = []
        errors = []
        
        for i, audio_file in enumerate(files):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    audio_file.save(temp_file.name)
                    
                    try:
                        embedding = voice_processor.extract_embedding(temp_file.name)
                        embeddings.append({
                            'index': i,
                            'filename': secure_filename(audio_file.filename),
                            'embedding': embedding
                        })
                    finally:
                        os.unlink(temp_file.name)
                        
            except Exception as e:
                errors.append({
                    'index': i,
                    'filename': secure_filename(audio_file.filename),
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'embeddings': embeddings,
            'errors': errors,
            'processed': len(embeddings),
            'failed': len(errors)
        })
        
    except Exception as e:
        logger.error(f"Error in batch_extract: {str(e)}")
        return jsonify({'error': 'Failed to process audio files'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

if __name__ == '__main__':
    port = int(os.environ.get('VOICE_PROCESSOR_PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
