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

## Transport Options

The MCP server supports multiple transport mechanisms:

### 1. stdio (Standard Input/Output)
Default transport for Claude Desktop integration.

**Use when:**
- Integrating with Claude Desktop
- Local development
- Direct process communication

### 2. HTTP with SSE (Server-Sent Events)
HTTP API with Server-Sent Events for streaming.

**Use when:**
- Remote access needed
- Multiple clients
- Web-based integrations
- RESTful API access

### 3. Both (Concurrent)
Run both transports simultaneously.

## Usage with Claude Desktop (stdio)

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
        "-T",
        "mcp-server",
        "python",
        "server_multi_transport.py"
      ],
      "env": {
        "VOICE_API_URL": "http://backend:8000",
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

## Usage via HTTP/SSE

The HTTP transport exposes a REST API on port 8001.

### Endpoints

**GET /tools** - List available tools
```bash
curl http://localhost:8001/tools
```

**POST /call-tool** - Call a tool (JSON response)
```bash
curl -X POST http://localhost:8001/call-tool \
  -H "Content-Type: application/json" \
  -d '{
    "name": "transcribe_audio",
    "arguments": {"audio_data": "BASE64_AUDIO"}
  }'
```

**POST /call-tool/sse** - Call a tool with SSE streaming
```bash
curl -N -X POST http://localhost:8001/call-tool/sse \
  -H "Content-Type: application/json" \
  -d '{
    "name": "synthesize_speech",
    "arguments": {"text": "Hello world"}
  }'
```

**GET /health** - Health check
```bash
curl http://localhost:8001/health
```

### Python Client Example

```python
import requests
import json

# Call tool via HTTP
response = requests.post(
    "http://localhost:8001/call-tool",
    json={
        "name": "check_voice_api_health",
        "arguments": {}
    }
)

result = response.json()
print(result)

# Stream with SSE
import sseclient

response = requests.post(
    "http://localhost:8001/call-tool/sse",
    json={
        "name": "synthesize_speech",
        "arguments": {"text": "Hello from HTTP!"}
    },
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    print(f"Event: {event.event}, Data: {event.data}")
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
- `MCP_TRANSPORT`: Transport mode - `stdio`, `http`, or `both` (default: stdio)
- `MCP_HTTP_PORT`: HTTP server port (default: 8001)
- `MCP_HTTP_HOST`: HTTP server host (default: 0.0.0.0)

## Running Standalone

### stdio mode (default)
```bash
pip install -r requirements.txt
export VOICE_API_URL=http://localhost:8000
export MCP_TRANSPORT=stdio
python server_multi_transport.py
```

### HTTP mode
```bash
pip install -r requirements.txt
export VOICE_API_URL=http://localhost:8000
export MCP_TRANSPORT=http
export MCP_HTTP_PORT=8001
python server_multi_transport.py
```

### Both modes concurrently
```bash
pip install -r requirements.txt
export VOICE_API_URL=http://localhost:8000
export MCP_TRANSPORT=both
python server_multi_transport.py
```

## Docker

### stdio mode (for Claude Desktop)
```bash
docker build -t voice-mcp-server .
docker run -e VOICE_API_URL=http://backend:8000 \
  -e MCP_TRANSPORT=stdio \
  voice-mcp-server
```

### HTTP mode (for remote access)
```bash
docker build -t voice-mcp-server .
docker run -p 8001:8001 \
  -e VOICE_API_URL=http://backend:8000 \
  -e MCP_TRANSPORT=http \
  voice-mcp-server
```

## Docker Compose

The `docker-compose.yml` includes both transports:

- `mcp-server`: stdio transport (for Claude Desktop)
- `mcp-server-http`: HTTP/SSE transport on port 8001 (for remote access)

```bash
# Start all services including both MCP transports
docker-compose up -d

# Access HTTP MCP server
curl http://localhost:8001/tools

# Use stdio MCP server with Claude Desktop
# (see Claude Desktop configuration above)
```

## Protocol

Uses Model Context Protocol (MCP) for communication with Claude agents.

### stdio Transport
- Input/Output: JSON-RPC over stdio
- Transport: Standard input/output streams
- Use case: Local Claude Desktop integration

### HTTP/SSE Transport
- Input/Output: JSON over HTTP
- Streaming: Server-Sent Events (SSE)
- Use case: Remote access, web integration, multiple clients

### Tools Schema
All transports expose the same MCP tools with identical schemas.
