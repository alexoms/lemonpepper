# MCP Server Transport Options

## Overview

The Voice Interaction MCP server now supports multiple transport mechanisms, allowing it to be used in various deployment scenarios beyond just Claude Desktop.

## Supported Transports

### 1. stdio (Standard Input/Output)

**Default transport for Claude Desktop integration.**

#### Characteristics:
- **Protocol:** JSON-RPC over stdio streams
- **Connection:** Direct process communication
- **Latency:** Very low (local)
- **Scalability:** Single client per instance
- **Security:** Process-level isolation

#### Use Cases:
- ✅ Claude Desktop integration
- ✅ Local agent development
- ✅ Command-line tools
- ✅ Process-to-process communication

#### Configuration:
```bash
export MCP_TRANSPORT=stdio
python server_multi_transport.py
```

#### Claude Desktop Config:
```json
{
  "mcpServers": {
    "voice-interaction": {
      "command": "docker",
      "args": ["compose", "-f", "/path/to/docker-compose.yml",
               "exec", "-T", "mcp-server", "python",
               "server_multi_transport.py"],
      "env": {
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

---

### 2. HTTP with SSE (Server-Sent Events)

**REST API with Server-Sent Events for web and remote integrations.**

#### Characteristics:
- **Protocol:** HTTP + JSON
- **Streaming:** Server-Sent Events (SSE)
- **Connection:** Network-based
- **Latency:** Low (network dependent)
- **Scalability:** Multiple concurrent clients
- **Security:** Standard HTTP security (add TLS, auth)

#### Use Cases:
- ✅ Remote agent access
- ✅ Web application integration
- ✅ Microservices architecture
- ✅ Load-balanced deployments
- ✅ Cross-platform clients
- ✅ REST API consumers

#### Configuration:
```bash
export MCP_TRANSPORT=http
export MCP_HTTP_PORT=8001
export MCP_HTTP_HOST=0.0.0.0
python server_multi_transport.py
```

#### API Endpoints:

**GET /tools**
```bash
curl http://localhost:14302/tools
```
Response: List of available MCP tools

**POST /call-tool**
```bash
curl -X POST http://localhost:14302/call-tool \
  -H "Content-Type: application/json" \
  -d '{"name": "check_voice_api_health", "arguments": {}}'
```
Response: JSON result

**POST /call-tool/sse**
```bash
curl -N -X POST http://localhost:14302/call-tool/sse \
  -H "Content-Type: application/json" \
  -d '{"name": "synthesize_speech", "arguments": {"text": "Hello"}}'
```
Response: SSE stream

**GET /health**
```bash
curl http://localhost:14302/health
```
Response: Server health status

#### Python Client Example:
```python
import requests

# Simple call
response = requests.post(
    "http://localhost:14302/call-tool",
    json={
        "name": "transcribe_audio",
        "arguments": {"audio_data": "BASE64_AUDIO"}
    }
)
print(response.json())

# With SSE streaming
import sseclient

response = requests.post(
    "http://localhost:14302/call-tool/sse",
    json={
        "name": "synthesize_speech",
        "arguments": {"text": "Hello world"}
    },
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    if event.event == "result":
        print(json.loads(event.data))
```

#### JavaScript Client Example:
```javascript
// Fetch API
async function callTool(name, arguments) {
  const response = await fetch('http://localhost:14302/call-tool', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, arguments})
  });
  return await response.json();
}

// EventSource for SSE
const eventSource = new EventSource(
  'http://localhost:14302/call-tool/sse?' +
  new URLSearchParams({
    name: 'check_voice_api_health',
    arguments: '{}'
  })
);

eventSource.addEventListener('result', (e) => {
  console.log('Result:', JSON.parse(e.data));
});

eventSource.addEventListener('complete', () => {
  eventSource.close();
});
```

---

### 3. Both (Concurrent)

**Run both stdio and HTTP transports simultaneously.**

#### Characteristics:
- All benefits of both transports
- Single server instance
- Shared resources and state
- Multiple access methods

#### Use Cases:
- ✅ Development environments
- ✅ Hybrid workflows
- ✅ Testing both interfaces
- ✅ Migration scenarios

#### Configuration:
```bash
export MCP_TRANSPORT=both
export MCP_HTTP_PORT=8001
python server_multi_transport.py
```

---

## Docker Compose Configuration

The `docker-compose.yml` includes separate services for each transport:

```yaml
services:
  # stdio transport (for Claude Desktop)
  mcp-server:
    environment:
      - MCP_TRANSPORT=stdio
    stdin_open: true
    tty: true
    command: ["python", "server_multi_transport.py"]

  # HTTP/SSE transport (for remote access)
  mcp-server-http:
    environment:
      - MCP_TRANSPORT=http
      - MCP_HTTP_PORT=8001
    ports:
      - "8001:8001"
    command: ["python", "server_multi_transport.py"]
```

Start both:
```bash
docker-compose up -d
```

Access:
- stdio: `docker-compose exec -T mcp-server python server_multi_transport.py`
- HTTP: `http://localhost:14302/tools`

---

## Comparison Matrix

| Feature | stdio | HTTP/SSE | Both |
|---------|-------|----------|------|
| **Latency** | Lowest | Low | Mixed |
| **Concurrent Clients** | 1 | Many | Many |
| **Remote Access** | ❌ | ✅ | ✅ |
| **Web Integration** | ❌ | ✅ | ✅ |
| **Claude Desktop** | ✅ | ⚠️ | ✅ |
| **Setup Complexity** | Low | Medium | Medium |
| **Resource Usage** | Low | Medium | High |
| **Authentication** | Process | HTTP Auth | Both |
| **Load Balancing** | ❌ | ✅ | ✅ |
| **Monitoring** | Limited | HTTP Logs | Both |

---

## Architecture Diagrams

### stdio Transport
```
┌──────────────┐
│Claude Desktop│
└──────┬───────┘
       │ stdin/stdout
       │ JSON-RPC
       ▼
┌──────────────┐
│  MCP Server  │
│   (stdio)    │
└──────┬───────┘
       │ HTTP
       ▼
┌──────────────┐
│ Voice API    │
│  Backend     │
└──────────────┘
```

### HTTP/SSE Transport
```
┌─────────┐  ┌─────────┐  ┌─────────┐
│Web App  │  │ Agent   │  │ Client  │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     │  HTTP/SSE (port 14302)  │
     └────────────┬────────────┘
                  ▼
         ┌────────────────┐
         │  MCP Server    │
         │   (HTTP/SSE)   │
         └────────┬───────┘
                  │ HTTP
                  ▼
         ┌────────────────┐
         │ Voice API      │
         │  Backend       │
         └────────────────┘
```

### Both Transports
```
┌──────────────┐         ┌─────────┐
│Claude Desktop│         │Web Clients│
└──────┬───────┘         └────┬─────┘
       │ stdio                │ HTTP/SSE
       └──────┬───────────────┘
              ▼
      ┌───────────────┐
      │  MCP Server   │
      │ (Both modes)  │
      └───────┬───────┘
              │ HTTP
              ▼
      ┌───────────────┐
      │  Voice API    │
      └───────────────┘
```

---

## Security Considerations

### stdio Transport
- ✅ Process isolation
- ✅ No network exposure
- ⚠️ Local access only
- 💡 Consider: File permission checks

### HTTP/SSE Transport
- ⚠️ Network exposed
- ❌ No authentication (default)
- ❌ No encryption (default)
- 💡 Consider: Add API keys, TLS, rate limiting

### Production Recommendations

#### For stdio:
1. Run as non-root user
2. Limit file system access
3. Monitor process spawning

#### For HTTP:
1. **Add authentication:**
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@http_app.post("/call-tool")
async def http_call_tool(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify token
    if not verify_token(credentials.credentials):
        raise HTTPException(401, "Invalid token")
    # ... rest of code
```

2. **Enable HTTPS:**
```bash
uvicorn server_multi_transport:http_app \
  --host 0.0.0.0 \
  --port 14302 \
  --ssl-keyfile key.pem \
  --ssl-certfile cert.pem
```

3. **Add rate limiting:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler

limiter = Limiter(key_func=get_remote_address)
http_app.state.limiter = limiter

@http_app.post("/call-tool")
@limiter.limit("10/minute")
async def http_call_tool(request: Request):
    # ... code
```

---

## Performance Tuning

### stdio Transport
- **Low overhead** - Direct process communication
- **Single client** - No connection pooling needed
- **Optimize:** Reduce tool complexity, cache results

### HTTP Transport
- **Connection pooling** - Reuse HTTP connections
- **Async processing** - Already using FastAPI async
- **Optimize:** Add caching, CDN for static responses

### Resource Limits
```yaml
# docker-compose.yml
services:
  mcp-server-http:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

---

## Monitoring & Logging

### stdio Transport
```python
# Access logs via stderr
logger.info(f"Tool called: {tool_name}")
```

### HTTP Transport
```python
# FastAPI access logs
# + Custom middleware
@http_app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url}")
    response = await call_next(request)
    return response
```

### Metrics
```python
# Add Prometheus metrics
from prometheus_client import Counter, Histogram

tool_calls = Counter('mcp_tool_calls', 'Tool calls', ['tool_name'])
tool_duration = Histogram('mcp_tool_duration', 'Tool duration')
```

---

## Migration Guide

### From stdio to HTTP

1. **Update client code:**
```python
# Before (stdio)
result = await mcp_client.call_tool("tool", args)

# After (HTTP)
response = requests.post(
    "http://server:8001/call-tool",
    json={"name": "tool", "arguments": args}
)
result = response.json()
```

2. **Update configuration:**
```bash
# Before
MCP_TRANSPORT=stdio

# After
MCP_TRANSPORT=http
MCP_HTTP_PORT=8001
```

3. **Test both (transition period):**
```bash
MCP_TRANSPORT=both
```

---

## Troubleshooting

### stdio Issues
**Problem:** "Broken pipe"
- Check process not terminated early
- Verify stdin/stdout not redirected

**Problem:** "Permission denied"
- Check Docker exec permissions
- Verify TTY allocation

### HTTP Issues
**Problem:** "Connection refused"
- Verify port 14302 exposed
- Check firewall rules
- Ensure MCP_TRANSPORT=http

**Problem:** "CORS errors"
- Check CORS middleware configured
- Verify origin in allow_origins

**Problem:** "Slow responses"
- Check network latency
- Monitor backend API response time
- Add connection pooling

---

## Future Enhancements

### Planned Transports
- ⏳ WebSocket bidirectional streaming
- ⏳ gRPC for high performance
- ⏳ GraphQL subscriptions
- ⏳ MQTT for IoT scenarios

### Planned Features
- ⏳ Authentication & authorization
- ⏳ Rate limiting & quotas
- ⏳ Request/response caching
- ⏳ Metrics & monitoring dashboard
- ⏳ Multi-region deployment
- ⏳ Load balancing support

---

## Resources

- **MCP Protocol:** https://modelcontextprotocol.io/
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **SSE Spec:** https://html.spec.whatwg.org/multipage/server-sent-events.html
- **Docker Compose:** https://docs.docker.com/compose/

---

**Transport flexibility enables the MCP server to adapt to your specific deployment needs!** 🚀
