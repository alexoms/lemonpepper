# Speech Demo - Experimental

This experimental project demonstrates real-time speech-to-text (STT) and text-to-speech (TTS) capabilities using the lemonpepper library components.

## Features

- **Speech-to-Text**: Real-time transcription using Whisper through WebSocket streaming
- **Text-to-Speech**: Natural voice synthesis using Picovoice Orca
- **Web Interface**: Modern React frontend with TypeScript
- **Streaming Audio**: Efficient audio processing and streaming

## Project Structure

```
experimental/
├── backend/          # Python FastAPI server
│   ├── server.py     # Main server with WebSocket and REST endpoints
│   └── requirements.txt
└── web/              # React frontend
    └── speech-demo/  # React TypeScript app
```

## Prerequisites

### Backend Requirements
- Python 3.8+
- Whisper model file (e.g., `ggml-base.en.bin`)
- Picovoice Access Key (for Orca TTS)

### Frontend Requirements
- Node.js 16+
- npm or yarn

## Setup Instructions

### Backend Setup

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

The API will be available at `http://localhost:8000`

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
REACT_APP_API_URL=http://localhost:8000
```

4. Start the development server:
```bash
npm start
```

The app will open at `http://localhost:3000`

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

### WebSocket
- `ws://localhost:8000/ws/stt` - Speech-to-text streaming

### REST
- `GET /` - API information
- `GET /health` - Health check
- `POST /api/tts/stream` - Text-to-speech synthesis

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
- Backend server is running on port 8000
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

## Future Enhancements

- Add voice activity detection (VAD)
- Support multiple languages
- Add recording history
- Implement audio visualization
- Add user authentication
- Support custom voice models
