# Voice Interaction MCP Integration for Claude Agent SDK

Complete guide for integrating voice capabilities into your Claude Agent SDK (formerly Claude Code SDK) applications.

## Overview

This MCP server provides voice interaction tools specifically designed for **Claude Agent SDK** applications. It enables your AI agents to:

- 🎤 **Transcribe speech** using Whisper (offline, privacy-friendly)
- 🔊 **Synthesize speech** using Picovoice Orca TTS
- 💬 **Handle voice conversations** end-to-end
- ✅ **Monitor service health** programmatically

**Why HTTP/SSE Transport?**
- Native support for remote agents and microservices
- Multiple concurrent agent instances
- Load balancing and horizontal scaling
- Standard HTTP tooling for monitoring and debugging
- RESTful API compatibility

## Quick Start

### 1. Deploy the MCP Server

```bash
cd experimental
./deploy.sh deploy
```

This starts:
- Voice API backend (port 14300)
- MCP server with HTTP/SSE transport (port 14302)
- Web demo interface (port 14301)

### 2. Configure Your Agent

Create or update `.mcp.json` in your Claude Agent SDK project root:

```json
{
  "mcpServers": {
    "voice-interaction": {
      "url": "http://localhost:14302",
      "transport": "http",
      "description": "Voice interaction tools for speech-to-text and text-to-speech"
    }
  }
}
```

**For remote deployment:**
```json
{
  "mcpServers": {
    "voice-interaction": {
      "url": "https://your-mcp-server.example.com",
      "transport": "http",
      "apiKey": "${VOICE_MCP_API_KEY}"
    }
  }
}
```

### 3. Verify Connection

```bash
# Check MCP server health
curl http://localhost:14302/health

# List available tools
curl http://localhost:14302/tools
```

## Available Tools

### 1. `transcribe_audio`

Convert speech to text using Whisper.

**Input:**
```typescript
{
  audio_data: string  // Base64-encoded PCM float32, 16kHz, mono
}
```

**Output:**
```typescript
{
  text: string  // Transcribed text
}
```

**Usage:**
```typescript
const transcription = await agent.callMCPTool(
  "voice-interaction",
  "transcribe_audio",
  { audio_data: base64AudioData }
);
```

### 2. `synthesize_speech`

Convert text to natural speech using Picovoice Orca.

**Input:**
```typescript
{
  text: string  // Text to synthesize
}
```

**Output:**
```typescript
{
  audio: string  // Base64-encoded WAV file
}
```

**Usage:**
```typescript
const speech = await agent.callMCPTool(
  "voice-interaction",
  "synthesize_speech",
  { text: "Hello, I am your AI assistant!" }
);
```

### 3. `voice_conversation`

Complete conversation loop: transcribe user input and synthesize agent response.

**Input:**
```typescript
{
  user_audio?: string,      // Base64-encoded user audio (optional)
  agent_response?: string   // Agent's text response to synthesize (optional)
}
```

**Output:**
```typescript
{
  user_text?: string,    // Transcribed user input
  agent_audio?: string   // Base64-encoded agent response audio
}
```

**Usage:**
```typescript
const result = await agent.callMCPTool(
  "voice-interaction",
  "voice_conversation",
  {
    user_audio: userAudioBase64,
    agent_response: "I understand. Let me help you with that."
  }
);
```

### 4. `check_voice_api_health`

Check the health status of voice services.

**Input:**
```typescript
{}  // No parameters
```

**Output:**
```typescript
{
  status: string,
  whisper_ready: boolean,
  orca_ready: boolean,
  version: string
}
```

## Integration Examples

### Example 1: Voice-Enabled Chat Agent (Python)

```python
from claude_agent_sdk import Agent
import base64
import pyaudio
import wave

class VoiceChatAgent:
    def __init__(self):
        self.agent = Agent()
        self.audio_config = {
            'format': pyaudio.paFloat32,
            'channels': 1,
            'rate': 16000,
            'chunk': 1024
        }

    def record_audio(self, duration=5):
        """Record audio from microphone"""
        p = pyaudio.PyAudio()
        stream = p.open(**self.audio_config, input=True)

        print(f"Recording for {duration} seconds...")
        frames = []
        for _ in range(0, int(self.audio_config['rate'] / self.audio_config['chunk'] * duration)):
            data = stream.read(self.audio_config['chunk'])
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = b''.join(frames)
        return base64.b64encode(audio_data).decode()

    def play_audio(self, base64_audio):
        """Play base64-encoded WAV audio"""
        audio_bytes = base64.b64decode(base64_audio)

        # Save temporarily
        with wave.open('/tmp/response.wav', 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_bytes)

        # Play audio (implementation depends on platform)
        import subprocess
        subprocess.run(['aplay', '/tmp/response.wav'])

    async def chat_loop(self):
        """Voice conversation loop"""
        print("Voice Chat Agent Started")

        while True:
            # Record user input
            user_audio = self.record_audio(duration=5)

            # Transcribe
            transcription = await self.agent.call_mcp_tool(
                "voice-interaction",
                "transcribe_audio",
                {"audio_data": user_audio}
            )

            user_text = transcription.get("text", "")
            print(f"User: {user_text}")

            if not user_text or "goodbye" in user_text.lower():
                print("Ending conversation...")
                break

            # Generate agent response
            response = await self.agent.generate_response(user_text)
            print(f"Agent: {response}")

            # Synthesize speech
            speech = await self.agent.call_mcp_tool(
                "voice-interaction",
                "synthesize_speech",
                {"text": response}
            )

            # Play response
            self.play_audio(speech.get("audio", ""))

# Run the agent
if __name__ == "__main__":
    import asyncio
    agent = VoiceChatAgent()
    asyncio.run(agent.chat_loop())
```

### Example 2: Voice-Enabled Agent (TypeScript)

```typescript
import { Agent } from '@anthropic-ai/agent-sdk';
import { spawn } from 'child_process';
import * as fs from 'fs';

class VoiceAgent {
  private agent: Agent;

  constructor() {
    this.agent = new Agent();
  }

  async captureAudio(duration: number = 5): Promise<string> {
    // Use SoX to record audio
    return new Promise((resolve, reject) => {
      const sox = spawn('sox', [
        '-d',                    // Default input device
        '-t', 'raw',            // Raw format
        '-b', '32',             // 32-bit
        '-e', 'floating-point', // Float
        '-r', '16000',          // 16kHz
        '-c', '1',              // Mono
        '-',                    // Output to stdout
        'trim', '0', `${duration}` // Duration
      ]);

      const chunks: Buffer[] = [];

      sox.stdout.on('data', (chunk) => {
        chunks.push(chunk);
      });

      sox.on('close', () => {
        const audioBuffer = Buffer.concat(chunks);
        const base64Audio = audioBuffer.toString('base64');
        resolve(base64Audio);
      });

      sox.on('error', reject);
    });
  }

  async playAudio(base64Audio: string): Promise<void> {
    // Decode and play audio
    const audioBuffer = Buffer.from(base64Audio, 'base64');
    const tempFile = '/tmp/agent_response.wav';

    fs.writeFileSync(tempFile, audioBuffer);

    return new Promise((resolve) => {
      const player = spawn('aplay', [tempFile]);
      player.on('close', () => resolve());
    });
  }

  async runVoiceConversation(): Promise<void> {
    console.log('Voice Agent Ready. Speak now...');

    while (true) {
      // Capture user speech
      const userAudio = await this.captureAudio(5);

      // Transcribe
      const transcription = await this.agent.callMCPTool(
        'voice-interaction',
        'transcribe_audio',
        { audio_data: userAudio }
      );

      const userText = transcription.text || '';
      console.log(`User: ${userText}`);

      if (!userText || userText.toLowerCase().includes('goodbye')) {
        console.log('Goodbye!');
        break;
      }

      // Generate response
      const response = await this.agent.generateResponse(userText);
      console.log(`Agent: ${response}`);

      // Synthesize
      const speech = await this.agent.callMCPTool(
        'voice-interaction',
        'synthesize_speech',
        { text: response }
      );

      // Play response
      await this.playAudio(speech.audio);
    }
  }
}

// Run
const agent = new VoiceAgent();
agent.runVoiceConversation().catch(console.error);
```

### Example 3: Multi-Agent Voice Workflow

```python
from claude_agent_sdk import Agent
import asyncio

class CoordinatorAgent:
    """Coordinates multiple agents with voice interaction"""

    def __init__(self):
        self.voice_agent = Agent(name="voice-handler")
        self.task_agent = Agent(name="task-executor")
        self.memory_agent = Agent(name="memory-keeper")

    async def process_voice_command(self, audio_base64):
        # Transcribe user command
        transcription = await self.voice_agent.call_mcp_tool(
            "voice-interaction",
            "transcribe_audio",
            {"audio_data": audio_base64}
        )

        user_command = transcription.get("text", "")

        # Store in memory
        await self.memory_agent.store_interaction(user_command)

        # Route to appropriate agent
        if "schedule" in user_command.lower():
            result = await self.task_agent.handle_scheduling(user_command)
        elif "remind" in user_command.lower():
            result = await self.task_agent.create_reminder(user_command)
        else:
            result = await self.task_agent.general_task(user_command)

        # Synthesize response
        response_audio = await self.voice_agent.call_mcp_tool(
            "voice-interaction",
            "synthesize_speech",
            {"text": result}
        )

        return {
            "user_input": user_command,
            "agent_response": result,
            "audio": response_audio.get("audio")
        }

# Usage
coordinator = CoordinatorAgent()
result = asyncio.run(coordinator.process_voice_command(user_audio))
```

## Advanced Configuration

### Custom MCP Server Endpoints

If you're running the MCP server on a different host/port:

```json
{
  "mcpServers": {
    "voice-interaction": {
      "url": "http://mcp-server.internal:8001",
      "transport": "http",
      "timeout": 60000
    }
  }
}
```

### With Authentication

Add API key authentication:

```json
{
  "mcpServers": {
    "voice-interaction": {
      "url": "https://voice-mcp.example.com",
      "transport": "http",
      "headers": {
        "Authorization": "Bearer ${VOICE_API_KEY}"
      }
    }
  }
}
```

### Multiple MCP Servers

Combine with other MCP servers:

```json
{
  "mcpServers": {
    "voice-interaction": {
      "url": "http://localhost:14302",
      "transport": "http"
    },
    "database": {
      "url": "http://localhost:8002",
      "transport": "http"
    },
    "web-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"]
    }
  }
}
```

## Audio Format Requirements

### For `transcribe_audio`

- **Format**: PCM float32 (32-bit floating point)
- **Sample Rate**: 16000 Hz (16 kHz)
- **Channels**: 1 (mono)
- **Encoding**: Base64

**Example conversion (Python):**
```python
import numpy as np
import base64

# Assuming you have audio as numpy array
audio_float32 = audio_data.astype(np.float32)
audio_bytes = audio_float32.tobytes()
audio_base64 = base64.b64encode(audio_bytes).decode()
```

### For `synthesize_speech`

**Output format:**
- **Format**: WAV file
- **Sample Rate**: 16000 Hz
- **Channels**: 1 (mono)
- **Bit Depth**: 16-bit PCM
- **Encoding**: Base64

## Best Practices

### 1. Error Handling

```python
async def safe_transcribe(agent, audio_data):
    try:
        result = await agent.call_mcp_tool(
            "voice-interaction",
            "transcribe_audio",
            {"audio_data": audio_data}
        )
        return result.get("text", "")
    except Exception as e:
        print(f"Transcription error: {e}")
        return None
```

### 2. Health Checks

```python
async def check_voice_services(agent):
    health = await agent.call_mcp_tool(
        "voice-interaction",
        "check_voice_api_health",
        {}
    )

    if not health.get("whisper_ready"):
        print("Warning: Whisper STT not ready")

    if not health.get("orca_ready"):
        print("Warning: Orca TTS not ready")

    return health.get("status") == "healthy"
```

### 3. Chunking Long Audio

For audio longer than 30 seconds, chunk it:

```python
def chunk_audio(audio_data, chunk_duration=10, sample_rate=16000):
    """Split audio into chunks"""
    chunk_size = chunk_duration * sample_rate
    audio_array = np.frombuffer(
        base64.b64decode(audio_data),
        dtype=np.float32
    )

    chunks = []
    for i in range(0, len(audio_array), chunk_size):
        chunk = audio_array[i:i + chunk_size]
        chunk_b64 = base64.b64encode(chunk.tobytes()).decode()
        chunks.append(chunk_b64)

    return chunks
```

### 4. Streaming Responses

For real-time feedback, use SSE streaming:

```python
import requests

response = requests.post(
    "http://localhost:14302/call-tool/sse",
    json={
        "name": "synthesize_speech",
        "arguments": {"text": "Long text to synthesize..."}
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b'data:'):
        # Process streaming data
        print(line.decode())
```

## Deployment Considerations

### Docker Deployment

The MCP server is containerized and can be deployed anywhere:

```yaml
# docker-compose.yml
services:
  mcp-server-http:
    image: voice-mcp-server:latest
    ports:
      - "14302:8001"
    environment:
      - MCP_TRANSPORT=http
      - MCP_HTTP_PORT=8001
      - VOICE_API_URL=http://backend:8000
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voice-mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: voice-mcp
  template:
    metadata:
      labels:
        app: voice-mcp
    spec:
      containers:
      - name: mcp-server
        image: voice-mcp-server:latest
        ports:
        - containerPort: 8001
        env:
        - name: MCP_TRANSPORT
          value: "http"
        - name: MCP_HTTP_PORT
          value: "8001"
---
apiVersion: v1
kind: Service
metadata:
  name: voice-mcp-service
spec:
  selector:
    app: voice-mcp
  ports:
  - port: 80
    targetPort: 8001
  type: LoadBalancer
```

## Troubleshooting

### Connection Issues

```bash
# Check if MCP server is running
curl http://localhost:14302/health

# Check if backend is accessible
curl http://localhost:14300/health

# View logs
docker-compose logs mcp-server-http
```

### Audio Quality Issues

- Ensure audio is 16kHz mono for best results
- Check microphone permissions
- Verify audio is not clipping (peak levels < 0 dB)

### Performance Optimization

- Use HTTP connection pooling
- Cache frequent TTS responses
- Consider GPU acceleration for Whisper
- Implement request queuing for high load

## Resources

- **MCP Specification**: https://modelcontextprotocol.io/
- **Claude Agent SDK Docs**: https://docs.anthropic.com/
- **Voice API Documentation**: See `experimental/README.md`
- **Transport Options**: See `experimental/MCP_TRANSPORTS.md`

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation
- Review API logs for errors

---

**Happy building with voice-enabled AI agents!** 🎤🤖
