# Quick Start Guide

Get the speech demo running in 5 minutes!

## Prerequisites

- Python 3.8+
- Node.js 16+
- Whisper model file (download from [whisper.cpp models](https://huggingface.co/ggerganov/whisper.cpp))
- Picovoice access key (get from [Picovoice Console](https://console.picovoice.ai/))

## Step 1: Start the Backend

```bash
# Navigate to backend
cd experimental/backend

# Install Python dependencies
pip install -r requirements.txt

# Set environment variables
export WHISPER_MODEL_PATH=/path/to/your/ggml-base.en.bin
export PICOVOICE_ACCESS_KEY=your_picovoice_access_key

# Start the server
python server.py
```

Backend will run on `http://localhost:8000`

## Step 2: Start the Frontend

Open a new terminal:

```bash
# Navigate to frontend
cd experimental/web/speech-demo

# Install Node dependencies (if not already done)
npm install

# Start React app
npm start
```

Frontend will open at `http://localhost:3000`

## Step 3: Test It Out

### Speech-to-Text
1. Click "Start Recording"
2. Allow microphone access when prompted
3. Speak clearly into your microphone
4. Watch the transcription appear in real-time
5. Click "Stop Recording" when done

### Text-to-Speech
1. Type some text in the textarea
2. Click "Speak"
3. Listen to the synthesized speech
4. Click "Stop Speaking" to interrupt

## Troubleshooting

### Backend won't start
- Make sure Whisper model file exists at the specified path
- Verify Picovoice access key is valid
- Check that port 8000 is available

### Frontend can't connect
- Verify backend is running on port 8000
- Check browser console for errors
- Try refreshing the page

### Microphone not working
- Grant microphone permissions in browser
- Use Chrome or Edge for best compatibility
- Make sure no other app is using the microphone

### No audio output
- Check system volume settings
- Verify speakers/headphones are connected
- Try a different browser

## Environment Variables

### Backend (.env)
```bash
WHISPER_MODEL_PATH=./models/ggml-base.en.bin
PICOVOICE_ACCESS_KEY=your_key_here
```

### Frontend (.env)
```bash
REACT_APP_API_URL=http://localhost:8000
```

## Next Steps

- Read the full [README.md](README.md) for architecture details
- Check backend [API documentation](http://localhost:8000/docs) (when running)
- Customize the UI in `web/speech-demo/src/App.tsx`
- Adjust audio settings in `backend/server.py`

## Resources

- [Whisper Models](https://huggingface.co/ggerganov/whisper.cpp)
- [Picovoice Console](https://console.picovoice.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
