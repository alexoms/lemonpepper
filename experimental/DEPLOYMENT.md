# Voice Interaction API - Deployment Guide

Complete guide for deploying the containerized voice interaction system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Docker Deployment](#docker-deployment)
5. [Using with Claude Agents](#using-with-claude-agents)
6. [API Documentation](#api-documentation)
7. [Troubleshooting](#troubleshooting)
8. [Production Considerations](#production-considerations)

## Prerequisites

### Required Software
- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 10GB disk space

### Required Files
1. **Picovoice Access Key**
   - Sign up at https://console.picovoice.ai/
   - Generate an access key
   - Free tier available

2. **Whisper Model**
   - Download from https://huggingface.co/ggerganov/whisper.cpp
   - Recommended: `ggml-base.en.bin` (150MB)
   - Place in `experimental/models/` directory

## Quick Start

### 1. Clone and Navigate
```bash
cd experimental
```

### 2. Create Environment File
```bash
cp .env.example .env
# Edit .env and add your PICOVOICE_ACCESS_KEY
nano .env
```

### 3. Download Whisper Model
```bash
mkdir -p models
cd models

# Download base English model (recommended)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

cd ..
```

### 4. Build and Start Services
```bash
docker-compose up -d
```

### 5. Verify Deployment
```bash
# Check services are running
docker-compose ps

# Check health
curl http://localhost:14300/health

# View logs
docker-compose logs -f
```

### 6. Access Services
- **Frontend**: http://localhost:14301
- **API Documentation**: http://localhost:14300/docs
- **OpenAPI Spec**: http://localhost:14300/openapi.json
- **Health Check**: http://localhost:14300/health

## Configuration

### Environment Variables

Create `.env` file in experimental directory:

```bash
# Required
PICOVOICE_ACCESS_KEY=your_key_here

# Optional (defaults shown)
WHISPER_MODEL_PATH=/app/models/ggml-base.en.bin
API_TIMEOUT=30
REACT_APP_API_URL=http://localhost:14300
```

### Model Selection

Different Whisper models available:

| Model | Size | Accuracy | Speed |
|-------|------|----------|-------|
| tiny.en | 75MB | Low | Fast |
| base.en | 150MB | Medium | Medium |
| small.en | 500MB | High | Slow |
| medium.en | 1.5GB | Very High | Very Slow |

Download and update `WHISPER_MODEL_PATH` accordingly.

## Docker Deployment

### Architecture

```
┌─────────────────────────────────────────┐
│         Docker Compose Stack            │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │ Frontend │  │ Backend  │  │ MCP  │ │
│  │ (nginx)  │  │ (FastAPI)│  │Server│ │
│  │  :3000   │  │  :8000   │  │      │ │
│  └────┬─────┘  └────┬─────┘  └──┬───┘ │
│       │             │             │     │
│       └─────────────┴─────────────┘     │
│          voice-interaction-network      │
└─────────────────────────────────────────┘
```

### Service Details

#### Backend
- **Image**: Custom Python 3.10
- **Port**: 8000
- **Volumes**: models directory
- **Dependencies**: Whisper, Orca TTS

#### Frontend
- **Image**: nginx:alpine
- **Port**: 3000 (mapped to 80 internal)
- **Built**: React production build
- **SPA**: Configured routing

#### MCP Server
- **Image**: Custom Python 3.10
- **Network**: Internal only
- **Protocol**: stdio via Docker exec

### Docker Commands

```bash
# Build services
docker-compose build

# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f [service]

# Restart specific service
docker-compose restart backend

# Rebuild and restart
docker-compose up -d --build

# Remove everything including volumes
docker-compose down -v
```

## Using with Claude Agents

### MCP Configuration

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "voice-interaction": {
      "command": "docker",
      "args": [
        "compose",
        "-f",
        "/absolute/path/to/experimental/docker-compose.yml",
        "exec",
        "-T",
        "mcp-server",
        "python",
        "server.py"
      ],
      "env": {
        "VOICE_API_URL": "http://backend:8000"
      }
    }
  }
}
```

### Available MCP Tools

1. **transcribe_audio**
   ```json
   {
     "audio_data": "base64_encoded_audio"
   }
   ```

2. **synthesize_speech**
   ```json
   {
     "text": "Text to speak"
   }
   ```

3. **voice_conversation**
   ```json
   {
     "user_audio": "base64_audio",
     "agent_response": "Response text"
   }
   ```

4. **check_voice_api_health**
   ```json
   {}
   ```

### Example Agent Workflow

```python
# User speaks via microphone
user_audio = capture_microphone()  # Returns base64

# Agent transcribes using MCP tool
transcription = mcp_call("transcribe_audio", {
    "audio_data": user_audio
})

# Agent processes and generates response
response = process_with_llm(transcription)

# Agent synthesizes speech response
audio = mcp_call("synthesize_speech", {
    "text": response
})

# Play to user
play_audio(decode_base64(audio))
```

## API Documentation

### REST Endpoints

#### GET /health
```bash
curl http://localhost:14300/health
```

Response:
```json
{
  "status": "healthy",
  "whisper_ready": true,
  "orca_ready": true,
  "version": "1.0.0"
}
```

#### POST /api/stt
```bash
curl -X POST http://localhost:14300/api/stt \
  -H "Content-Type: application/json" \
  -d '{"audio_data": "base64_audio_here"}'
```

#### POST /api/tts
```bash
curl -X POST http://localhost:14300/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  --output speech.wav
```

#### POST /api/tts/sse (Server-Sent Events)
```bash
curl -N -X POST http://localhost:14300/api/tts/sse \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

### WebSocket Endpoint

#### ws://localhost:8000/ws/stt

Send:
```json
{
  "type": "audio",
  "data": "base64_audio"
}
```

Receive:
```json
{
  "type": "transcription",
  "text": "transcribed text",
  "is_final": false
}
```

### Interactive Documentation

Visit http://localhost:14300/docs for Swagger UI with:
- Full API reference
- Interactive testing
- Request/response schemas
- Code examples

## Troubleshooting

### Backend Won't Start

**Issue**: Backend container exits immediately

**Solution**:
```bash
# Check logs
docker-compose logs backend

# Verify model file exists
ls -lh experimental/models/

# Ensure environment variables are set
docker-compose exec backend env | grep WHISPER

# Rebuild container
docker-compose build --no-cache backend
docker-compose up -d backend
```

### Frontend Can't Connect to Backend

**Issue**: CORS errors or connection refused

**Solution**:
```bash
# Ensure backend is running
docker-compose ps backend

# Check network connectivity
docker-compose exec frontend ping backend

# Verify API URL in frontend container
docker-compose exec frontend env | grep API_URL
```

### MCP Server Connection Issues

**Issue**: Claude can't connect to MCP server

**Solution**:
1. Verify absolute path in config
2. Ensure Docker daemon is running
3. Test MCP server manually:
```bash
docker-compose exec -T mcp-server python server.py
```

### Audio Quality Issues

**Issue**: Poor transcription accuracy or TTS quality

**Solution**:
- Use higher quality Whisper model (base → small → medium)
- Ensure audio is 16kHz, mono, PCM float32
- Check microphone quality
- Reduce background noise

### Performance Issues

**Issue**: Slow transcription or synthesis

**Solution**:
```bash
# Allocate more resources to Docker
# Docker Desktop → Settings → Resources

# Use smaller Whisper model
# Edit .env: WHISPER_MODEL_PATH=/app/models/ggml-tiny.en.bin

# Check resource usage
docker stats
```

## Production Considerations

### Security

1. **Enable Authentication**
   - Add API key authentication
   - Use OAuth2 for frontend
   - Implement rate limiting

2. **HTTPS/TLS**
   - Use reverse proxy (nginx, Traefik)
   - Obtain SSL certificates (Let's Encrypt)
   - Update CORS origins

3. **Network Security**
   - Use internal networks for inter-service communication
   - Expose only necessary ports
   - Implement firewall rules

### Scaling

1. **Horizontal Scaling**
   ```yaml
   backend:
     deploy:
       replicas: 3
   ```

2. **Load Balancing**
   - Add nginx/HAProxy for load balancing
   - Use Docker Swarm or Kubernetes

3. **Caching**
   - Add Redis for session management
   - Cache common TTS phrases

### Monitoring

1. **Logging**
   ```yaml
   logging:
     driver: "json-file"
     options:
       max-size: "10m"
       max-file: "3"
   ```

2. **Metrics**
   - Prometheus for metrics collection
   - Grafana for visualization
   - Health check endpoints

3. **Alerts**
   - Set up alerting for service failures
   - Monitor resource usage
   - Track API response times

### Backup

```bash
# Backup configuration
tar -czf voice-api-config.tar.gz experimental/.env experimental/models/

# Backup volumes
docker run --rm -v experimental_models:/data -v $(pwd):/backup \
  alpine tar czf /backup/models-backup.tar.gz /data
```

### Updates

```bash
# Pull latest changes
git pull origin feature/web

# Rebuild containers
docker-compose build --pull

# Rolling update (no downtime)
docker-compose up -d --no-deps --build backend
docker-compose up -d --no-deps --build frontend
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/alexoms/lemonpepper/issues
- Documentation: See README.md files in each directory
- API Docs: http://localhost:14300/docs

## License

MIT License - See LICENSE file for details
