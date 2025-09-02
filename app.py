from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import warnings
import tempfile
import os
import base64

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Enhanced create_embedding function
def create_embedding(audio_data, prompt_text=None):
    try:
        # Load audio data using librosa
        if isinstance(audio_data, str):
            # If audio_data is a file path
            y, sr = librosa.load(audio_data, sr=16000)
        else:
            # If audio_data is already numpy array
            y = audio_data
            sr = 16000
        
        # Extract features (MFCC as a simple embedding)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Take mean across time axis to get fixed-size embedding
        embedding = np.mean(mfccs, axis=1)
        
        # If prompt_text is provided, you could incorporate it here
        # For now, we'll just use audio features
        
        return {
            "embedding": embedding.tolist(),
            "shape": embedding.shape,
            "prompt_text": prompt_text
        }
    except Exception as e:
        return {"error": f"Embedding creation failed: {str(e)}"}

def cosine_similarity(vec1, vec2):
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    except Exception as e:
        print(f"Cosine similarity error: {e}")
        return 0.0

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "service": "voice-embedding"})

@app.route('/create-embedding', methods=['POST'])
def handle_create_embedding():
    try:
        # Handle file upload
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            result = create_embedding(temp_file.name)
        
        # Clean up
        os.unlink(temp_file.name)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New endpoint for text-dependent verification
@app.route('/create-text-dependent-embedding', methods=['POST'])
def handle_create_text_dependent_embedding():
    try:
        # Handle file upload
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Get prompt text from form data or JSON
        prompt_text = request.form.get('prompt_text') or request.json.get('prompt_text') if request.json else None
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            result = create_embedding(temp_file.name, prompt_text)
        
        # Clean up
        os.unlink(temp_file.name)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Base64 endpoint for serverless compatibility
@app.route('/create-text-dependent-embedding-base64', methods=['POST'])
def handle_create_text_dependent_embedding_base64():
    try:
        data = request.get_json()
        
        if not data or 'audio_base64' not in data:
            return jsonify({"error": "No audio_base64 data provided"}), 400
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(data['audio_base64'])
        except Exception as e:
            return jsonify({"error": f"Invalid base64 audio data: {str(e)}"}), 400
        
        prompt_text = data.get('prompt_text')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            result = create_embedding(temp_file.name, prompt_text)
        
        # Clean up
        os.unlink(temp_file.name)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/compare-embeddings', methods=['POST'])
def handle_compare_embeddings():
    try:
        data = request.get_json()
        
        if not data or 'embedding1' not in data or 'embedding2' not in data:
            return jsonify({"error": "Two embeddings required"}), 400
        
        embedding1 = data['embedding1']
        embedding2 = data['embedding2']
        
        # Calculate similarity
        similarity = cosine_similarity(embedding1, embedding2)
        
        return jsonify({
            "similarity": similarity,
            "match": similarity > 0.8  # Threshold for match
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
