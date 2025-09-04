const express = require('express');
const multer = require('multer');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');
const axios = require('axios');
const FormData = require('form-data');
const ffmpeg = require('fluent-ffmpeg');
const winston = require('winston');

const app = express();

// Configure logging
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.printf(({ timestamp, level, message, ...meta }) => {
      return `${timestamp} - ${level.toUpperCase()} - ${message} ${
        Object.keys(meta).length ? JSON.stringify(meta) : ''
      }`;
    })
  ),
  transports: [new winston.transports.Console()],
});

// Middleware
app.use(express.json({ limit: '8mb' }));
app.use(express.urlencoded({ extended: true, limit: '8mb' }));

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: {
    fileSize: 8 * 1024 * 1024, // 8MB
    files: 1,
  },
  fileFilter: (req, file, cb) => {
    const allowedMimes = [
      'audio/wav',
      'audio/wave',
      'audio/x-wav',
      'audio/mpeg',
      'audio/mp3',
      'audio/ogg',
      'audio/webm',
      'audio/flac',
      'audio/x-flac',
    ];

    if (
      allowedMimes.includes(file.mimetype) ||
      file.originalname.match(/\.(wav|mp3|ogg|webm|flac)$/i)
    ) {
      cb(null, true);
    } else {
      cb(
        new Error(
          'Invalid audio format. Supported formats: WAV, MP3, OGG, WebM, FLAC'
        ),
        false
      );
    }
  },
});

// Voice processor configuration
const VOICE_CONFIG = {
  MIN_AUDIO_LENGTH: 1.0,
  MAX_AUDIO_LENGTH: 10.0,
  VERIFICATION_THRESHOLD: 0.75,
  SAMPLE_RATE: 16000,
  MAX_EMBEDDINGS_CACHE: 100,
  MAX_COMPARISONS: 20,
  BATCH_SIZE: 5,
};

// Python model service URL (if using external Python service)
const PYTHON_SERVICE_URL =
  process.env.PYTHON_SERVICE_URL || 'http://localhost:5000';

class VoiceProcessor {
  constructor() {
    this.embeddingCache = new Map();
    this.tempDir = os.tmpdir();
  }

  async getMemoryUsage() {
    const used = process.memoryUsage();
    return {
      rss_mb: Math.round((used.rss / 1024 / 1024) * 100) / 100,
      heap_used_mb: Math.round((used.heapUsed / 1024 / 1024) * 100) / 100,
      heap_total_mb: Math.round((used.heapTotal / 1024 / 1024) * 100) / 100,
      external_mb: Math.round((used.external / 1024 / 1024) * 100) / 100,
    };
  }

  async convertToWav(inputBuffer, originalName) {
    return new Promise((resolve, reject) => {
      const inputPath = path.join(
        this.tempDir,
        `input_${Date.now()}_${Math.random()
          .toString(36)
          .substr(2, 9)}.${path.extname(originalName)}`
      );
      const outputPath = path.join(
        this.tempDir,
        `output_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.wav`
      );

      // Write input buffer to temporary file
      fs.writeFile(inputPath, inputBuffer)
        .then(() => {
          ffmpeg(inputPath)
            .toFormat('wav')
            .audioCodec('pcm_s16le')
            .audioFrequency(VOICE_CONFIG.SAMPLE_RATE)
            .audioChannels(1)
            .on('end', async () => {
              try {
                const wavBuffer = await fs.readFile(outputPath);
                // Cleanup temp files
                await this.cleanupTempFile(inputPath);
                await this.cleanupTempFile(outputPath);
                resolve(wavBuffer);
              } catch (err) {
                reject(err);
              }
            })
            .on('error', async (err) => {
              await this.cleanupTempFile(inputPath);
              await this.cleanupTempFile(outputPath);
              reject(err);
            })
            .save(outputPath);
        })
        .catch(reject);
    });
  }

  async cleanupTempFile(filePath) {
    try {
      await fs.unlink(filePath);
    } catch (err) {
      // Ignore errors - file might already be deleted
    }
  }

  async extractEmbeddingViaPython(audioBuffer, originalName) {
    try {
      // Convert to WAV if needed
      let wavBuffer = audioBuffer;
      if (!originalName.toLowerCase().endsWith('.wav')) {
        logger.info(`Converting ${originalName} to WAV format`);
        wavBuffer = await this.convertToWav(audioBuffer, originalName);
      }

      // Create form data for the Python service
      const formData = new FormData();
      formData.append('audio', wavBuffer, 'audio.wav');

      // Call Python service
      const response = await axios.post(
        `${PYTHON_SERVICE_URL}/extract_embedding`,
        formData,
        {
          headers: {
            ...formData.getHeaders(),
            'Content-Type': 'multipart/form-data',
          },
          timeout: 30000, // 30 second timeout
        }
      );

      if (response.data.success) {
        return response.data.embedding;
      } else {
        throw new Error(response.data.error || 'Failed to extract embedding');
      }
    } catch (error) {
      if (error.response) {
        throw new Error(error.response.data.error || 'Python service error');
      }
      throw error;
    }
  }

  async extractEmbeddingViaLocal(audioBuffer, originalName) {
    // This would require a local Python installation with the necessary libraries
    return new Promise((resolve, reject) => {
      const tempPath = path.join(
        this.tempDir,
        `temp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.wav`
      );

      fs.writeFile(tempPath, audioBuffer)
        .then(() => {
          // Create a Python script execution
          const pythonScript = `
import sys
import json
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

try:
    encoder = VoiceEncoder()
    wav = preprocess_wav('${tempPath}')
    embedding = encoder.embed_utterance(wav)
    result = {
        'success': True,
        'embedding': embedding.tolist()
    }
    print(json.dumps(result))
except Exception as e:
    result = {
        'success': False,
        'error': str(e)
    }
    print(json.dumps(result))
`;

          const python = spawn('python3', ['-c', pythonScript]);
          let output = '';
          let error = '';

          python.stdout.on('data', (data) => {
            output += data.toString();
          });

          python.stderr.on('data', (data) => {
            error += data.toString();
          });

          python.on('close', async (code) => {
            await this.cleanupTempFile(tempPath);

            if (code !== 0) {
              reject(new Error(`Python script failed: ${error}`));
              return;
            }

            try {
              const result = JSON.parse(output.trim());
              if (result.success) {
                resolve(result.embedding);
              } else {
                reject(new Error(result.error));
              }
            } catch (parseError) {
              reject(
                new Error(
                  `Failed to parse Python output: ${parseError.message}`
                )
              );
            }
          });
        })
        .catch(reject);
    });
  }

  async extractEmbedding(audioBuffer, originalName) {
    try {
      logger.info(`Extracting embedding from ${originalName}`);

      // Try Python service first, fallback to local if configured
      //   if (process.env.USE_LOCAL_PYTHON === 'true') {
      return await this.extractEmbeddingViaLocal(audioBuffer, originalName);
      //   } else {
      //     return await this.extractEmbeddingViaPython(audioBuffer, originalName);
      //   }
    } catch (error) {
      logger.error(`Error extracting embedding: ${error.message}`);
      throw error;
    }
  }

  compareEmbeddings(embedding1, embedding2) {
    try {
      // Validate inputs
      if (!Array.isArray(embedding1) || !Array.isArray(embedding2)) {
        throw new Error('Embeddings must be arrays');
      }

      if (embedding1.length !== embedding2.length) {
        throw new Error('Embeddings must have same length');
      }

      // Calculate cosine similarity
      let dotProduct = 0;
      let norm1 = 0;
      let norm2 = 0;

      for (let i = 0; i < embedding1.length; i++) {
        dotProduct += embedding1[i] * embedding2[i];
        norm1 += embedding1[i] * embedding1[i];
        norm2 += embedding2[i] * embedding2[i];
      }

      norm1 = Math.sqrt(norm1);
      norm2 = Math.sqrt(norm2);

      if (norm1 === 0 || norm2 === 0) {
        return 0.0;
      }

      return dotProduct / (norm1 * norm2);
    } catch (error) {
      logger.error(`Error comparing embeddings: ${error.message}`);
      throw error;
    }
  }

  verifyVoice(
    testEmbedding,
    enrolledEmbeddings,
    threshold = VOICE_CONFIG.VERIFICATION_THRESHOLD
  ) {
    try {
      if (!enrolledEmbeddings || enrolledEmbeddings.length === 0) {
        throw new Error('No enrolled embeddings provided');
      }

      // Limit number of comparisons
      let embeddingsToCompare = enrolledEmbeddings;
      if (enrolledEmbeddings.length > VOICE_CONFIG.MAX_COMPARISONS) {
        logger.warn(
          `Too many enrollments (${enrolledEmbeddings.length}), using first ${VOICE_CONFIG.MAX_COMPARISONS}`
        );
        embeddingsToCompare = enrolledEmbeddings.slice(
          0,
          VOICE_CONFIG.MAX_COMPARISONS
        );
      }

      const similarities = [];

      // Process in batches
      for (
        let i = 0;
        i < embeddingsToCompare.length;
        i += VOICE_CONFIG.BATCH_SIZE
      ) {
        const batch = embeddingsToCompare.slice(i, i + VOICE_CONFIG.BATCH_SIZE);

        for (const enrolledEmbedding of batch) {
          const similarity = this.compareEmbeddings(
            testEmbedding,
            enrolledEmbedding
          );
          similarities.push(similarity);
        }

        // Force garbage collection periodically
        if (i % (VOICE_CONFIG.BATCH_SIZE * 2) === 0 && global.gc) {
          global.gc();
        }
      }

      // Calculate statistics
      const avgSimilarity =
        similarities.reduce((sum, sim) => sum + sim, 0) / similarities.length;
      const maxSimilarity = Math.max(...similarities);
      const isMatch = avgSimilarity >= threshold;

      return {
        is_match: isMatch,
        confidence: avgSimilarity,
        max_confidence: maxSimilarity,
        threshold: threshold,
        num_comparisons: similarities.length,
      };
    } catch (error) {
      logger.error(`Error in voice verification: ${error.message}`);
      throw error;
    }
  }

  cleanup() {
    // Clear embedding cache if it gets too large
    if (this.embeddingCache.size > VOICE_CONFIG.MAX_EMBEDDINGS_CACHE) {
      this.embeddingCache.clear();
      logger.info('Cleared embedding cache');
    }

    // Force garbage collection if available
    if (global.gc) {
      global.gc();
    }
  }
}

// Initialize processor
const voiceProcessor = new VoiceProcessor();

// Middleware for memory monitoring
const memoryMonitor = (req, res, next) => {
  const startTime = Date.now();
  const initialMemory = process.memoryUsage();

  res.on('finish', async () => {
    const endTime = Date.now();
    const finalMemory = process.memoryUsage();
    const duration = endTime - startTime;

    logger.info(
      `${req.method} ${req.path} - ${
        res.statusCode
      } - ${duration}ms - Memory: ${Math.round(
        initialMemory.heapUsed / 1024 / 1024
      )}MB -> ${Math.round(finalMemory.heapUsed / 1024 / 1024)}MB`
    );

    // Cleanup if memory usage is high
    if (finalMemory.heapUsed > 1000 * 1024 * 1024) {
      // 1GB
      logger.warn(
        `High memory usage: ${Math.round(finalMemory.heapUsed / 1024 / 1024)}MB`
      );
      voiceProcessor.cleanup();
    }
  });

  next();
};

app.use(memoryMonitor);

// Routes
app.get('/', (req, res) => {
  res.json({
    service: 'Voice Biometrics Processor',
    status: 'online',
    version: '1.0.0',
    platform: 'nodejs',
  });
});

app.get('/health', async (req, res) => {
  try {
    const memoryUsage = await voiceProcessor.getMemoryUsage();

    res.json({
      status: 'healthy',
      service: 'voice_processor_nodejs',
      ...memoryUsage,
      platform: 'nodejs',
      uptime: process.uptime(),
      node_version: process.version,
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      error: error.message,
    });
  }
});

app.post('/extract_embedding', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No audio file provided' });
    }

    const { buffer, originalname, size } = req.file;

    // Validate file size
    if (size > 8 * 1024 * 1024) {
      return res
        .status(400)
        .json({ error: 'File too large. Maximum size is 8MB' });
    }

    logger.info(`Processing audio file: ${originalname}, size: ${size} bytes`);

    const embedding = await voiceProcessor.extractEmbedding(
      buffer,
      originalname
    );

    res.json({
      success: true,
      embedding: embedding,
      embedding_size: embedding.length,
    });
  } catch (error) {
    logger.error(`Error in extract_embedding: ${error.message}`);

    if (
      error.message.includes('too short') ||
      error.message.includes('too long')
    ) {
      return res.status(400).json({ error: error.message });
    }

    res.status(500).json({ error: 'Failed to process audio', message: error });
  } finally {
    // Cleanup
    voiceProcessor.cleanup();
  }
});

app.post('/compare_embeddings', (req, res) => {
  try {
    const { embedding1, embedding2 } = req.body;

    if (!embedding1 || !embedding2) {
      return res.status(400).json({ error: 'Both embeddings required' });
    }

    const similarity = voiceProcessor.compareEmbeddings(embedding1, embedding2);

    res.json({
      success: true,
      similarity: similarity,
      is_match: similarity >= VOICE_CONFIG.VERIFICATION_THRESHOLD,
    });
  } catch (error) {
    logger.error(`Error in compare_embeddings: ${error.message}`);

    if (
      error.message.includes('must be arrays') ||
      error.message.includes('same length')
    ) {
      return res.status(400).json({ error: error.message });
    }

    res.status(500).json({ error: 'Failed to compare embeddings' });
  }
});

app.post('/verify_voice', (req, res) => {
  try {
    const { test_embedding, enrolled_embeddings, threshold } = req.body;

    if (!test_embedding || !enrolled_embeddings) {
      return res
        .status(400)
        .json({ error: 'Test embedding and enrolled embeddings required' });
    }

    if (
      !Array.isArray(enrolled_embeddings) ||
      enrolled_embeddings.length === 0
    ) {
      return res.status(400).json({ error: 'No enrolled embeddings provided' });
    }

    const result = voiceProcessor.verifyVoice(
      test_embedding,
      enrolled_embeddings,
      threshold
    );

    res.json({
      success: true,
      ...result,
    });
  } catch (error) {
    logger.error(`Error in verify_voice: ${error.message}`);

    if (error.message.includes('No enrolled embeddings')) {
      return res.status(400).json({ error: error.message });
    }

    res.status(500).json({ error: 'Failed to verify voice' });
  }
});

app.get('/memory_status', async (req, res) => {
  try {
    const memoryUsage = await voiceProcessor.getMemoryUsage();

    res.json({
      ...memoryUsage,
      uptime: process.uptime(),
      platform: 'nodejs',
      node_version: process.version,
      cache_size: voiceProcessor.embeddingCache.size,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/cleanup', async (req, res) => {
  try {
    const beforeMemory = process.memoryUsage();

    voiceProcessor.cleanup();

    // Force garbage collection if available
    if (global.gc) {
      global.gc();
    }

    const afterMemory = process.memoryUsage();

    res.json({
      success: true,
      memory_before_mb: Math.round(beforeMemory.heapUsed / 1024 / 1024),
      memory_after_mb: Math.round(afterMemory.heapUsed / 1024 / 1024),
      cache_cleared: true,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Error handlers
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res
        .status(413)
        .json({ error: 'File too large. Maximum size is 8MB.' });
    }
  }

  logger.error(`Unhandled error: ${error.message}`);
  res.status(500).json({ error: 'Internal server error' });
});

app.use('*', (req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('Received SIGTERM, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.info('Received SIGINT, shutting down gracefully');
  process.exit(0);
});

// Start server
const PORT = process.env.PORT || 8000;

app.listen(PORT, '0.0.0.0', () => {
  logger.info(`Voice Biometrics Processor (Node.js) starting on port ${PORT}`);
  logger.info(
    `Memory usage: ${Math.round(
      process.memoryUsage().heapUsed / 1024 / 1024
    )}MB`
  );

  if (process.env.USE_LOCAL_PYTHON === 'true') {
    logger.info('Using local Python installation for model inference');
  } else {
    logger.info(`Using Python service at: ${PYTHON_SERVICE_URL}`);
  }
});

module.exports = app;
