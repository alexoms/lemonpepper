import React, { useState, useRef, useEffect } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_BASE_URL = API_BASE_URL.replace('http', 'ws');

function App() {
  // Speech-to-Text state
  const [isRecording, setIsRecording] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [sttStatus, setSttStatus] = useState('Ready');

  // Text-to-Speech state
  const [ttsText, setTtsText] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [ttsStatus, setTtsStatus] = useState('Ready');

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioElementRef = useRef<HTMLAudioElement>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Speech-to-Text Functions
  const startRecording = async () => {
    try {
      setSttStatus('Initializing...');

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });

      mediaStreamRef.current = stream;

      // Create audio context
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      const source = audioContextRef.current.createMediaStreamSource(stream);

      // Create processor for audio data
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      // Connect WebSocket
      const ws = new WebSocket(`${WS_BASE_URL}/ws/stt`);
      wsRef.current = ws;

      ws.onopen = () => {
        setSttStatus('Connected - Recording...');
        setIsRecording(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'transcription') {
            setTranscription((prev) => prev + ' ' + data.text);
          } else if (data.error) {
            setSttStatus(`Error: ${data.error}`);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setSttStatus('WebSocket error');
        stopRecording();
      };

      ws.onclose = () => {
        setSttStatus('Disconnected');
        setIsRecording(false);
      };

      // Process audio data
      processor.onaudioprocess = (e) => {
        if (ws.readyState === WebSocket.OPEN) {
          const audioData = e.inputBuffer.getChannelData(0);
          // Convert to base64
          const buffer = new ArrayBuffer(audioData.length * 4);
          const view = new Float32Array(buffer);
          view.set(audioData);
          const base64Audio = arrayBufferToBase64(buffer);

          ws.send(JSON.stringify({
            type: 'audio',
            data: base64Audio
          }));
        }
      };

      source.connect(processor);
      processor.connect(audioContextRef.current.destination);

    } catch (error) {
      console.error('Error starting recording:', error);
      setSttStatus(`Error: ${error}`);
    }
  };

  const stopRecording = () => {
    // Send stop message
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop' }));
      wsRef.current.close();
    }

    // Stop audio processing
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    // Stop media stream
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setIsRecording(false);
    setSttStatus('Stopped');
  };

  // Text-to-Speech Functions
  const speakText = async () => {
    if (!ttsText.trim()) {
      setTtsStatus('Please enter text to speak');
      return;
    }

    try {
      setIsSpeaking(true);
      setTtsStatus('Synthesizing...');

      const response = await fetch(`${API_BASE_URL}/api/tts/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: ttsText }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Get audio blob
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);

      // Play audio
      if (audioElementRef.current) {
        audioElementRef.current.src = audioUrl;
        audioElementRef.current.onplay = () => setTtsStatus('Speaking...');
        audioElementRef.current.onended = () => {
          setTtsStatus('Finished');
          setIsSpeaking(false);
          URL.revokeObjectURL(audioUrl);
        };
        audioElementRef.current.onerror = (e) => {
          setTtsStatus('Playback error');
          setIsSpeaking(false);
          URL.revokeObjectURL(audioUrl);
        };
        await audioElementRef.current.play();
      }

    } catch (error) {
      console.error('Error with TTS:', error);
      setTtsStatus(`Error: ${error}`);
      setIsSpeaking(false);
    }
  };

  const stopSpeaking = () => {
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.currentTime = 0;
    }
    setIsSpeaking(false);
    setTtsStatus('Stopped');
  };

  // Helper function to convert ArrayBuffer to base64
  const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Speech Demo</h1>
        <p>Whisper STT & Picovoice Orca TTS</p>
      </header>

      <div className="container">
        {/* Speech-to-Text Section */}
        <div className="section stt-section">
          <h2>Speech to Text</h2>
          <div className="status">Status: {sttStatus}</div>

          <div className="controls">
            {!isRecording ? (
              <button
                className="btn btn-primary"
                onClick={startRecording}
              >
                Start Recording
              </button>
            ) : (
              <button
                className="btn btn-danger"
                onClick={stopRecording}
              >
                Stop Recording
              </button>
            )}
          </div>

          <div className="transcription-box">
            <h3>Transcription:</h3>
            <div className="transcription-text">
              {transcription || 'Start recording to see transcription...'}
            </div>
            <button
              className="btn btn-secondary"
              onClick={() => setTranscription('')}
            >
              Clear
            </button>
          </div>
        </div>

        {/* Text-to-Speech Section */}
        <div className="section tts-section">
          <h2>Text to Speech</h2>
          <div className="status">Status: {ttsStatus}</div>

          <div className="controls">
            <textarea
              className="tts-input"
              value={ttsText}
              onChange={(e) => setTtsText(e.target.value)}
              placeholder="Enter text to synthesize..."
              rows={5}
            />
          </div>

          <div className="controls">
            {!isSpeaking ? (
              <button
                className="btn btn-primary"
                onClick={speakText}
                disabled={!ttsText.trim()}
              >
                Speak
              </button>
            ) : (
              <button
                className="btn btn-danger"
                onClick={stopSpeaking}
              >
                Stop Speaking
              </button>
            )}
          </div>

          <audio ref={audioElementRef} style={{ display: 'none' }} />
        </div>
      </div>
    </div>
  );
}

export default App;
