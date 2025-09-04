# Use Node.js LTS with Alpine for smaller image size
FROM node:18-alpine

# Install system dependencies for audio processing
RUN apk add --no-cache \
    ffmpeg \
    python3 \
    py3-pip \
    build-base \
    python3-dev \
    linux-headers \
    && ln -sf python3 /usr/bin/python

# Install Python dependencies for local model inference (optional)
# RUN pip3 install --no-cache-dir \
#     numpy==1.24.3 \
#     scipy==1.10.1 \
#     scikit-learn==1.3.0 \
#     librosa==0.10.1 \
#     resemblyzer==0.1.2 \
#     soundfile==0.12.1

# Create app directory
WORKDIR /usr/src/app

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy application code
COPY app.js ./

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nodejs -u 1001

# Change ownership of the working directory
RUN chown -R nodejs:nodejs /usr/src/app
USER nodejs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:8000/health', (res) => { \
    process.exit(res.statusCode === 200 ? 0 : 1) \
  }).on('error', () => process.exit(1))"

# Start the application
CMD ["node", "app.js"]