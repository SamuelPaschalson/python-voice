from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:3001", "http://your-frontend-domain.com"])

# Increase payload size limit
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enhanced create_embedding function
def create_embedding(audio_data, prompt_text=None):
    """Create voice embedding from audio data with text dependency"""
    try:
        # Save audio to temporary file
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name

        try:
            # Load audio file
            y, sr = librosa.load(tmp_path, sr=16000, duration=3.0)

            # Check if audio is long enough
            if len(y) < sr * 0.5:
                return {"error": "Audio too short"}

            # Pre-emphasis to amplify high frequencies
            y = librosa.effects.preemphasis(y)

            # Extract MFCC features with more parameters
            mfccs = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=20,
                n_fft=2048,
                hop_length=512,
                n_mels=128
            )

            # Calculate delta and delta-delta features
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

            # Combine all features
            features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])

            # Calculate statistics
            feature_mean = np.mean(features, axis=1)
            feature_std = np.std(features, axis=1)
            feature_max = np.max(features, axis=1)
            feature_min = np.min(features, axis=1)

            # Combine to create embedding
            embedding = np.concatenate([feature_mean, feature_std, feature_max, feature_min])

            return embedding.tolist()
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

    except Exception as e:
        return {"error": f"Embedding creation failed: {str(e)}"}

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    try:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return (similarity + 1) / 2
    except:
        return 0.0

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "service": "voice-embedding"})

@app.route('/create-embedding', methods=['POST'])
def handle_create_embedding():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        audio_data = audio_file.read()

        embedding = create_embedding(audio_data)

        if 'error' in embedding:
            return jsonify(embedding), 400

        return jsonify({"embedding": embedding})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New endpoint for text-dependent verification
@app.route('/create-text-dependent-embedding', methods=['POST'])
def handle_create_text_dependent_embedding():
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        
        # Check file size
        audio_file.seek(0, 2)  # Seek to end
        file_size = audio_file.tell()
        audio_file.seek(0)  # Reset to beginning
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            return jsonify({"error": "File too large. Maximum size is 10MB"}), 400
        
        audio_data = audio_file.read()
        prompt_text = request.form.get('prompt_text', 'default')

        embedding = create_embedding(audio_data, prompt_text)

        if 'error' in embedding:
            return jsonify(embedding), 400

        return jsonify({
            "embedding": embedding,
            "prompt_text": prompt_text
        })

    except Exception as e:
        print(f"Error in create-text-dependent-embedding: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/compare-embeddings', methods=['POST'])
def handle_compare_embeddings():
    try:
        data = request.json
        embedding1 = data.get('embedding1')
        embedding2 = data.get('embedding2')
        threshold = data.get('threshold', 0.7)

        if not embedding1 or not embedding2:
            return jsonify({"error": "Both embeddings are required"}), 400

        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        similarity = cosine_similarity(emb1, emb2)

        return jsonify({
            "similarity": float(similarity),
            "match": similarity >= threshold,
            "threshold": threshold
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
