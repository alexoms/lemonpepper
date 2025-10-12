# Voice Interaction MCP Server

Model Context Protocol (MCP) server that exposes voice interaction capabilities to Claude agents.

## Overview

This MCP server provides tools for:
- **Speech-to-Text**: Transcribe audio using Whisper
- **Text-to-Speech**: Synthesize speech using Picovoice Orca
- **Voice Conversations**: Complete voice interaction loops
- **Health Checks**: Monitor API availability

## Tools Exposed

### 1. transcribe_audio
Transcribe audio data to text using Whisper.

**Input:**
- `audio_data`: Base64-encoded PCM float32 audio at 16kHz

**Output:**
- Transcribed text

### 2. synthesize_speech
Convert text to speech using Picovoice Orca TTS.

**Input:**
- `text`: Text to synthesize

**Output:**
- Base64-encoded WAV audio

### 3. voice_conversation
Complete voice interaction: transcribe + synthesize.

**Input:**
- `user_audio`: Base64-encoded user audio (optional)
- `agent_response`: Agent's text response (optional)

**Output:**
- Transcription and synthesized response

### 4. check_voice_api_health
Check Voice API health status.

**Output:**
- Health status of STT and TTS services

## Usage with Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "voice-interaction": {
      "command": "docker",
      "args": [
        "compose",
        "-f",
        "/path/to/experimental/docker-compose.yml",
        "exec",
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

## Usage in Agentic Workflows

Example agent workflow:

```python
# User speaks
user_audio_b64 = capture_microphone()

# Agent uses MCP tool to transcribe
result = await mcp_client.call_tool(
    "transcribe_audio",
    {"audio_data": user_audio_b64}
)

# Agent processes and generates response
agent_response = process_user_input(result)

# Agent synthesizes response
audio = await mcp_client.call_tool(
    "synthesize_speech",
    {"text": agent_response}
)

# Play audio to user
play_audio(audio)
```

## Environment Variables

- `VOICE_API_URL`: Voice API base URL (default: http://backend:8000)
- `API_TIMEOUT`: HTTP request timeout in seconds (default: 30)

## Running Standalone

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export VOICE_API_URL=http://localhost:8000

# Run server
python server.py
```

## Docker

```bash
# Build
docker build -t voice-mcp-server .

# Run
docker run -e VOICE_API_URL=http://backend:8000 voice-mcp-server
```

## Protocol

Uses Model Context Protocol (MCP) for communication with Claude agents.
- Input/Output: JSON-RPC over stdio
- Transport: Standard input/output streams
- Tools: Defined in MCP tool schema format
