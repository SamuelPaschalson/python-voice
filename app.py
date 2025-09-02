from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import warnings
import tempfile
import os
import base64
import soundfile as sf

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Fixed create_embedding function
def create_embedding(audio_data, prompt_text=None):
    try:
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            tmp_filename = tmp_file.name
        
        # Load audio file using librosa
        y, sr = librosa.load(tmp_filename, sr=16000)  # Resample to 16kHz
        
        # Clean up temporary file
        os.unlink(tmp_filename)
        
        # Extract audio features (MFCCs as a simple example)
        # In a real implementation, you'd use a proper voice embedding model
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Average across time to get a fixed-length vector
        embedding = np.mean(mfccs, axis=1)
        
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return {"embedding": embedding.tolist(), "status": "success"}
    
    except Exception as e:
        return {"error": f"Embedding creation failed: {str(e)}"}

def cosine_similarity(vec1, vec2):
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
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
        
        result = create_embedding(audio_data)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Fixed text-dependent verification endpoint
@app.route('/create-text-dependent-embedding', methods=['POST'])
def handle_create_text_dependent_embedding():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        if 'prompt_text' not in request.form:
            return jsonify({"error": "No prompt text provided"}), 400
        
        audio_file = request.files['audio']
        prompt_text = request.form['prompt_text']
        audio_data = audio_file.read()
        
        # Call the actual embedding function
        result = create_embedding(audio_data, prompt_text)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify({
            "embedding": result["embedding"],
            "prompt_text": prompt_text,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Fixed Base64 endpoint for serverless compatibility
@app.route('/create-text-dependent-embedding-base64', methods=['POST'])
def handle_create_text_dependent_embedding_base64():
    try:
        data = request.get_json()
        
        if not data or 'audio_base64' not in data:
            return jsonify({"error": "No base64 audio data provided"}), 400
        
        if 'prompt_text' not in data:
            return jsonify({"error": "No prompt text provided"}), 400
        
        audio_base64 = data['audio_base64']
        prompt_text = data['prompt_text']
        
        # Decode base64 audio
        audio_data = base64.b64decode(audio_base64)
        
        # Call the actual embedding function
        result = create_embedding(audio_data, prompt_text)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify({
            "embedding": result["embedding"],
            "prompt_text": prompt_text,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/compare-embeddings', methods=['POST'])
def handle_compare_embeddings():
    try:
        data = request.get_json()
        
        if not data or 'embedding1' not in data or 'embedding2' not in data:
            return jsonify({"error": "Both embeddings are required"}), 400
        
        embedding1 = data['embedding1']
        embedding2 = data['embedding2']
        
        similarity = cosine_similarity(embedding1, embedding2)
        
        return jsonify({
            "similarity": float(similarity),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
