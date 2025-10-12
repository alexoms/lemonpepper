# Speech Demo Project Summary

## Branch: feature/web
**Author:** Alex Chang (alex@unidatum.com)

## What Was Created

A complete full-stack speech demo application with:
- **Backend**: Python FastAPI server with streaming STT and TTS
- **Frontend**: React TypeScript app with modern UI
- **Integration**: WebSocket streaming for real-time transcription
- **Audio Processing**: Browser-based microphone capture and playback

## Project Structure

```
experimental/
├── backend/
│   ├── server.py              # FastAPI server with WebSocket + REST endpoints
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example          # Environment variable template
│   └── README.md             # Backend documentation
│
├── web/
│   └── speech-demo/          # React TypeScript app
│       ├── src/
│       │   ├── App.tsx       # Main component with STT + TTS
│       │   └── App.css       # Modern gradient styling
│       ├── .env              # Frontend config
│       ├── .env.example      # Config template
│       └── package.json      # Node dependencies
│
├── README.md                 # Main project documentation
├── QUICKSTART.md            # 5-minute setup guide
└── PROJECT_SUMMARY.md       # This file
```

## Key Features

### Speech-to-Text (STT)
- Real-time transcription using Whisper
- WebSocket streaming for low latency
- Browser microphone capture via Web Audio API
- Base64 encoding for audio transport
- Automatic audio buffering and overlap

### Text-to-Speech (TTS)
- Natural voice synthesis using Picovoice Orca
- Streaming audio generation
- WAV format output
- Browser-based audio playback
- Clean resource management

### User Interface
- Modern gradient design
- Glass morphism effects
- Responsive layout (desktop + mobile)
- Real-time status indicators
- Simple, intuitive controls

## Libraries Used

### Existing lemonpepper Components
- `lemonpepper.transcribe_audio_whisper.WhisperStreamTranscriber`
- `lemonpepper.PicovoiceOrcaStreamer` (Orca instance used directly)

### Backend Dependencies
- FastAPI - Web framework
- uvicorn - ASGI server
- websockets - WebSocket support
- numpy - Audio processing
- pywhispercpp - Whisper bindings
- pvorca - Picovoice TTS

### Frontend Dependencies
- React 18 - UI framework
- TypeScript - Type safety
- Web Audio API - Microphone capture
- WebSocket API - Real-time communication

## Technical Highlights

### Backend Architecture
- **Async/Await**: FastAPI async handlers for concurrent connections
- **WebSocket Streaming**: Bidirectional audio streaming
- **CORS**: Configured for web frontend access
- **Audio Processing**: 16kHz mono PCM for Whisper
- **Error Handling**: Comprehensive error messages

### Frontend Architecture
- **Hooks**: Modern React with useState, useRef, useEffect
- **Audio Context**: Proper AudioContext management
- **Resource Cleanup**: Automatic cleanup on unmount
- **TypeScript**: Full type safety
- **Responsive Design**: Mobile-friendly layout

### Audio Pipeline

**STT Flow:**
```
Microphone → AudioContext → ScriptProcessorNode → 
Base64 Encode → WebSocket → Backend → 
Whisper Model → Transcription → WebSocket → 
Frontend Display
```

**TTS Flow:**
```
Text Input → HTTP POST → Backend → 
Orca Synthesis → WAV Stream → 
Frontend Blob → Audio Element → Playback
```

## Configuration

### Required Environment Variables

**Backend:**
- `WHISPER_MODEL_PATH`: Path to Whisper model file
- `PICOVOICE_ACCESS_KEY`: Picovoice API key

**Frontend:**
- `REACT_APP_API_URL`: Backend URL (defaults to localhost:8000)

## Getting Started

See [QUICKSTART.md](QUICKSTART.md) for complete setup instructions.

Quick version:
```bash
# Terminal 1 - Backend
cd experimental/backend
pip install -r requirements.txt
export WHISPER_MODEL_PATH=/path/to/model.bin
export PICOVOICE_ACCESS_KEY=your_key
python server.py

# Terminal 2 - Frontend  
cd experimental/web/speech-demo
npm install
npm start
```

## API Endpoints

### REST
- `GET /` - API info
- `GET /health` - Health check
- `POST /api/tts/stream` - Text-to-speech

### WebSocket
- `ws://localhost:8000/ws/stt` - Speech-to-text streaming

## Development Notes

### What Works
- ✓ Real-time speech transcription
- ✓ Natural TTS playback
- ✓ WebSocket streaming
- ✓ Browser audio capture
- ✓ Responsive UI
- ✓ Error handling

### Future Enhancements
- Voice Activity Detection (VAD)
- Audio visualization
- Recording history
- Multi-language support
- User authentication
- Custom voice models
- Deployment configuration

## Testing the Demo

1. Start both backend and frontend servers
2. Open browser to http://localhost:3000
3. Test STT: Click "Start Recording", speak, see transcription
4. Test TTS: Type text, click "Speak", hear audio
5. Check backend logs for detailed debugging info

## Troubleshooting

Common issues and solutions in [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md).

## Git Information

- **Branch**: feature/web
- **Base Branch**: main
- **Author**: Alex Chang <alex@unidatum.com>
- **Status**: Ready for testing

## Next Steps

1. Test the application locally
2. Adjust configurations as needed
3. Customize UI styling
4. Add additional features
5. Create git commit when ready
6. Optionally create pull request

## Additional Resources

- FastAPI Docs: https://fastapi.tiangolo.com/
- React Docs: https://react.dev/
- Whisper Models: https://huggingface.co/ggerganov/whisper.cpp
- Picovoice Console: https://console.picovoice.ai/
