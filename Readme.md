Voice Biometrics Processor (Node.js)
A Node.js-based voice biometrics processor that provides speaker verification and voice embedding extraction capabilities. This service maintains the same API endpoints as the original Python Flask implementation while leveraging trained online models.

Features
Voice Embedding Extraction: Extract unique voice embeddings from audio files
Speaker Verification: Compare voice embeddings to verify speaker identity
Multiple Audio Formats: Support for WAV, MP3, OGG, WebM, and FLAC
Memory Optimization: Built-in memory monitoring and cleanup
Scalable Architecture: Supports both local Python inference and external model services
RESTful API: Same endpoints as the original Flask implementation
Architecture
The Node.js implementation provides two modes for accessing trained voice models:

External Python Service Mode (Recommended): Communicates with a separate Python service that hosts the trained models
Local Python Mode: Executes Python scripts locally (requires Python environment)
Installation
Prerequisites
Node.js 16+ and npm
FFmpeg (for audio conversion)
Python 3.8+ with voice processing libraries (for local mode)
Quick Start
bash

# Clone the repository

git clone <repository-url>
cd voice-biometrics-nodejs

# Install Node.js dependencies

npm install

# Install system dependencies (Ubuntu/Debian)

sudo apt-get update
sudo apt-get install ffmpeg python3 python3-pip

# Install Python dependencies (for local mode)

pip3 install numpy scipy scikit-learn librosa resemblyzer soundfile

# Start the server

npm start
Docker Deployment
bash

# Build the Docker image

docker build -t voice-biometrics-nodejs .

# Run the container

docker run -p 8000:8000 voice-biometrics-nodejs
Railway Deployment
Connect your GitHub repository to Railway
Set environment variables:
USE_LOCAL_PYTHON=true (for local Python mode)
PYTHON_SERVICE_URL=<your-python-service-url> (for external service mode)
Deploy using the provided railway.json configuration
Configuration
Environment Variables
Variable Description Default
PORT Server port 8000
NODE_ENV Environment (production/staging/development) development
USE_LOCAL_PYTHON Use local Python installation false
PYTHON_SERVICE_URL External Python service URL http://localhost:5000
Voice Processing Configuration
javascript
const VOICE_CONFIG = {
MIN_AUDIO_LENGTH: 1.0, // Minimum audio length in seconds
MAX_AUDIO_LENGTH: 10.0, // Maximum audio length in seconds
VERIFICATION_THRESHOLD: 0.75, // Voice verification threshold
SAMPLE_RATE: 16000, // Audio sample rate
MAX_EMBEDDINGS_CACHE: 100, // Maximum cached embeddings
MAX_COMPARISONS: 20, // Maximum enrollment comparisons
BATCH_SIZE: 5 // Batch processing size
};
API Endpoints
Health Check
bash
GET /
GET /health
Extract Voice Embedding
bash
POST /extract_embedding
Content-Type: multipart/form-data

# Form data:

audio: <audio_file>
Response:

json
{
"success": true,
"embedding": [0.123, -0.456, ...],
"
}
