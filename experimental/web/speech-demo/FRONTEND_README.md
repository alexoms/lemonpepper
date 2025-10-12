# Speech Demo Frontend

React TypeScript application for demonstrating speech-to-text and text-to-speech capabilities.

## Features

- Real-time microphone capture and streaming
- WebSocket-based speech-to-text
- Text-to-speech with audio playback
- Modern, responsive UI with gradient styling
- Audio context management with proper cleanup

## Installation

```bash
npm install
```

## Configuration

Create a `.env` file (optional):

```bash
REACT_APP_API_URL=http://localhost:8000
```

If not specified, defaults to `http://localhost:8000`.

## Development

Start the development server:

```bash
npm start
```

Opens at `http://localhost:3000`

## Building for Production

```bash
npm run build
```

Creates optimized production build in the `build` folder.

## Component Structure

### App.tsx
Main component containing:
- Speech-to-text controls and display
- Text-to-speech input and playback
- WebSocket management
- Audio context handling

### App.css
Styling with:
- Gradient background
- Glass morphism effects
- Responsive layout
- Button animations

## Audio Pipeline

### Speech-to-Text
1. Request microphone access via `getUserMedia`
2. Create AudioContext at 16kHz sample rate
3. Use ScriptProcessorNode to capture audio chunks
4. Convert audio to base64 and send via WebSocket
5. Receive and display transcription results

### Text-to-Speech
1. Send text to backend via POST request
2. Receive WAV audio blob
3. Create object URL and play via HTMLAudioElement
4. Clean up resources after playback

## Browser Compatibility

Recommended browsers:
- Chrome 80+
- Edge 80+
- Firefox 90+
- Safari 14+

Requires:
- Web Audio API support
- WebSocket support
- MediaStream API support

## Microphone Permissions

The app requires microphone access. In Chrome/Edge:
1. Click the lock icon in the address bar
2. Allow microphone access
3. Refresh the page if needed

## Troubleshooting

### "Microphone not available"
- Check browser permissions
- Ensure no other app is using the microphone
- Try a different browser

### "WebSocket connection failed"
- Verify backend server is running
- Check firewall settings
- Ensure correct API URL in `.env`

### "Audio won't play"
- Check browser audio settings
- Verify speakers/headphones are connected
- Try a different browser

## Development Notes

### Audio Context Cleanup
The app properly cleans up audio resources:
- Closes audio context on unmount
- Stops media streams when recording stops
- Disconnects audio processors

### WebSocket Management
- Automatic reconnection not implemented (feature for future)
- Proper connection state handling
- Error messages displayed in UI

### State Management
Uses React hooks for state:
- `useState` for UI state
- `useRef` for audio resources
- `useEffect` for cleanup

## Future Enhancements

- Add audio visualization (waveform/spectrum)
- Implement voice activity detection
- Add recording history
- Support for multiple languages
- Keyboard shortcuts
- Download transcription as text file
