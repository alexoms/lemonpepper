# Voice Interaction API - Experimental

Complete voice interaction system with speech-to-text, text-to-speech, and MCP integration for AI agents.

## 🚀 Quick Start with Docker

```bash
cd experimental

# Setup environment
cp .env.example .env
# Edit .env and add your PICOVOICE_ACCESS_KEY

# Deploy everything
./deploy.sh deploy

# Access services
# Frontend:  http://localhost:14301
# API Docs:  http://localhost:14300/docs
# Health:    http://localhost:14300/health
```

That's it! 🎉

## Features

### Core Capabilities
- **Speech-to-Text**: Real-time transcription using Whisper
- **Text-to-Speech**: Natural voice synthesis using Picovoice Orca
- **Multiple Protocols**: REST, WebSocket, and SSE streaming
- **Web Interface**: Modern React frontend with TypeScript
- **MCP Integration**: Tools for Claude agents to enable voice interactions
- **Docker Deployment**: Fully containerized with docker-compose

### API Features
- ✅ REST endpoints for synchronous operations
- ✅ WebSocket for bidirectional streaming
- ✅ Server-Sent Events (SSE) for audio streaming
- ✅ OpenAPI/Swagger documentation
- ✅ Health monitoring
- ✅ CORS configured for web access

### MCP Tools for Agents
- `transcribe_audio` - Convert speech to text
- `synthesize_speech` - Convert text to speech
- `voice_conversation` - Complete interaction loop
- `check_voice_api_health` - Service status

## Project Structure

```
experimental/
├── backend/              # Python FastAPI server
│   ├── server_enhanced.py   # Enhanced server with REST, WS, SSE
│   ├── Dockerfile
│   └── requirements.txt
├── web/speech-demo/      # React TypeScript frontend
│   ├── src/
│   ├── Dockerfile
│   └── nginx.conf
├── mcp-server/           # MCP server for Claude agents
│   ├── server.py
│   ├── Dockerfile
│   └── requirements.txt
├── models/               # Whisper models directory
├── docker-compose.yml    # Orchestration
├── deploy.sh            # Deployment script
├── DEPLOYMENT.md        # Full deployment guide
└── .env.example         # Environment template
```

## Prerequisites

### For Docker Deployment (Recommended)
- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- Picovoice Access Key ([Get Free Key](https://console.picovoice.ai/))
- Whisper model (auto-downloaded by deploy script)

### For Manual Setup
- Python 3.10+
- Node.js 18+
- Whisper model file
- Picovoice Access Key

## Deployment Options

### Option 1: Docker (Recommended) 🐳

**One-command deployment:**

```bash
cd experimental
./deploy.sh deploy
```

The script will:
1. Check prerequisites
2. Setup environment
3. Download Whisper model
4. Build containers
5. Start all services

**Other commands:**
```bash
./deploy.sh start    # Start services
./deploy.sh stop     # Stop services
./deploy.sh logs     # View logs
./deploy.sh status   # Check status
./deploy.sh help     # Show all commands
```

### Option 2: Manual Setup

#### Backend Setup

1. Navigate to the backend directory:
```bash
cd experimental/backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export WHISPER_MODEL_PATH=/path/to/your/whisper/model.bin
export PICOVOICE_ACCESS_KEY=your_picovoice_access_key
```

4. Run the server:
```bash
python server.py
```

The API will be available at `http://localhost:14300`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd experimental/web/speech-demo
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file (optional, defaults to localhost:8000):
```bash
REACT_APP_API_URL=http://localhost:14300
```

4. Start the development server:
```bash
npm start
```

The app will open at `http://localhost:14301`

## Usage

### Speech-to-Text
1. Click "Start Recording" to begin capturing audio from your microphone
2. Speak into your microphone
3. Watch the transcription appear in real-time
4. Click "Stop Recording" to end the session
5. Use "Clear" to reset the transcription

### Text-to-Speech
1. Enter text in the textarea
2. Click "Speak" to synthesize and play the audio
3. Click "Stop Speaking" to interrupt playback

## API Endpoints

Full interactive documentation available at http://localhost:14300/docs

### Speech-to-Text
- `POST /api/stt` - REST endpoint (single audio chunk)
- `POST /api/stt/stream` - SSE streaming
- `ws://localhost:14300/ws/stt` - WebSocket streaming

### Text-to-Speech
- `POST /api/tts` - REST endpoint (returns WAV file)
- `POST /api/tts/sse` - SSE streaming (audio chunks)

### Information
- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Swagger UI
- `GET /openapi.json` - OpenAPI specification

## Technical Details

### Backend
- **Framework**: FastAPI with WebSocket support
- **STT Engine**: Whisper (via pywhispercpp)
- **TTS Engine**: Picovoice Orca
- **Audio Format**: 16kHz, mono, PCM float32 (STT) / int16 (TTS)

### Frontend
- **Framework**: React 18 with TypeScript
- **Audio Capture**: Web Audio API with MediaStream
- **WebSocket**: Native WebSocket API
- **Styling**: Custom CSS with gradient backgrounds

## Libraries Used

This demo leverages the existing lemonpepper library components:
- `lemonpepper.transcribe_audio_whisper.WhisperStreamTranscriber` - Whisper STT
- `lemonpepper.PicovoiceOrcaStreamer.PicovoiceOrcaStreamer` - Orca TTS

## Troubleshooting

### Microphone Access
Ensure your browser has permission to access the microphone. The app requires HTTPS in production or localhost for microphone access.

### WebSocket Connection
If the WebSocket fails to connect, verify:
- Backend server is running on port 14300
- CORS is properly configured
- Firewall allows WebSocket connections

### Audio Issues
- Check browser console for detailed error messages
- Verify audio drivers are working
- Test with different browsers (Chrome/Edge recommended)

## Development Notes

- The backend uses streaming for efficient real-time processing
- Audio data is base64-encoded for WebSocket transmission
- The frontend includes proper cleanup of audio resources
- Both components are designed for low-latency operation

## Using with Claude Agents

This API is exposed via MCP (Model Context Protocol) for agentic workflows.

### MCP Configuration

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "voice-interaction": {
      "command": "docker",
      "args": [
        "compose", "-f", "/path/to/experimental/docker-compose.yml",
        "exec", "-T", "mcp-server", "python", "server.py"
      ]
    }
  }
}
```

### Example Agent Workflow

```python
# User speaks to agent
user_audio = capture_microphone()

# Agent transcribes
text = mcp.call_tool("transcribe_audio", {
    "audio_data": user_audio
})

# Agent processes and responds
response = agent.process(text)

# Agent speaks back
audio = mcp.call_tool("synthesize_speech", {
    "text": response
})

play_audio(audio)
```

See `mcp-server/README.md` for full MCP documentation.

## Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Complete deployment guide
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute quick start
- **[backend/README.md](backend/README.md)** - Backend API details
- **[mcp-server/README.md](mcp-server/README.md)** - MCP integration
- **API Docs**: http://localhost:14300/docs (when running)

## Future Enhancements

- Voice activity detection (VAD)
- Multi-language support
- Recording history and playback
- Audio visualization
- User authentication and rate limiting
- Custom voice models
- Kubernetes deployment configs
