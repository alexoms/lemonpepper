# Voice Interaction API - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────┐        ┌─────────────────────────────┐ │
│  │   Web Browser          │        │   Claude Desktop App        │ │
│  │   (Human Users)        │        │   (AI Agent)                │ │
│  │                        │        │                             │ │
│  │  - Microphone input    │        │  - MCP tool calls           │ │
│  │  - Audio playback      │        │  - Agentic workflows        │ │
│  │  - Real-time UI        │        │  - Voice interactions       │ │
│  └───────────┬────────────┘        └──────────────┬──────────────┘ │
│              │                                     │                │
└──────────────┼─────────────────────────────────────┼────────────────┘
               │                                     │
               │ HTTP/WS                             │ stdio
               ▼                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Docker Compose Network                          │
│                   (voice-interaction-network)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌──────────┐ │
│  │    Frontend         │   │      Backend        │   │   MCP    │ │
│  │    Container        │   │      Container      │   │  Server  │ │
│  │                     │   │                     │   │          │ │
│  │  ┌───────────────┐ │   │  ┌───────────────┐ │   │  ┌────┐  │ │
│  │  │    nginx      │ │   │  │   FastAPI     │ │   │  │MCP │  │ │
│  │  │   :80 → :3000 │◄┼───┼──┤   Server      │◄┼───┼──┤Srv │  │ │
│  │  │               │ │   │  │   :8000       │ │   │  └────┘  │ │
│  │  │  React SPA    │ │   │  │               │ │   │          │ │
│  │  │  Build files  │ │   │  │  REST API     │ │   │  Tools:  │ │
│  │  └───────────────┘ │   │  │  WebSocket    │ │   │  • STT   │ │
│  │                     │   │  │  SSE Stream   │ │   │  • TTS   │ │
│  │  Health: /health   │   │  │               │ │   │  • Conv  │ │
│  │  Logs: access.log  │   │  │  ┌─────────┐ │ │   │  • Health│ │
│  └─────────────────────┘   │  │  │Whisper  │ │ │   └──────────┘ │
│                             │  │  │  STT    │ │ │                │
│  Volume:                    │  │  └─────────┘ │ │                │
│  ./build → nginx root       │  │  ┌─────────┐ │ │                │
│                             │  │  │ Orca    │ │ │                │
│                             │  │  │  TTS    │ │ │                │
│                             │  │  └─────────┘ │ │                │
│                             │  │               │ │                │
│                             │  │  Health:      │ │                │
│                             │  │  /health      │ │                │
│                             │  │  /docs        │ │                │
│                             │  └───────────────┘ │                │
│                             │                     │                │
│                             │  Volume:            │                │
│                             │  ./models (RO)      │                │
│                             │  ├─ ggml-base.en    │                │
│                             │  └─ ...             │                │
│                             └─────────────────────┘                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### Frontend Container

**Technology Stack:**
- Base: nginx:alpine
- Build: Multi-stage (node:18-alpine → nginx)
- Framework: React 18 + TypeScript

**Responsibilities:**
- Serve React SPA
- Handle browser routing
- Provide health endpoint
- Optimize static assets (gzip, caching)

**Communication:**
- Outbound: HTTP/WebSocket to backend
- Inbound: HTTP from browsers

**Ports:**
- External: 3000
- Internal: 80

### Backend Container

**Technology Stack:**
- Base: Python 3.10-slim
- Framework: FastAPI + Uvicorn
- STT: Whisper (pywhispercpp)
- TTS: Picovoice Orca

**Responsibilities:**
- Transcribe audio to text (STT)
- Synthesize text to speech (TTS)
- Serve REST API
- Handle WebSocket connections
- Stream via SSE
- Provide OpenAPI documentation

**Endpoints:**

| Path | Method | Type | Purpose |
|------|--------|------|---------|
| / | GET | REST | API info |
| /health | GET | REST | Health check |
| /docs | GET | REST | Swagger UI |
| /openapi.json | GET | REST | OpenAPI spec |
| /api/stt | POST | REST | Transcribe audio |
| /api/stt/stream | POST | SSE | Stream transcription |
| /api/tts | POST | REST | Synthesize speech |
| /api/tts/sse | POST | SSE | Stream synthesis |
| /ws/stt | WS | WebSocket | Bidirectional STT |

**Ports:**
- External: 8000
- Internal: 8000

**Volumes:**
- ./models → /app/models (read-only)

### MCP Server Container

**Technology Stack:**
- Base: Python 3.10-slim
- Protocol: Model Context Protocol
- Transport: stdio

**Responsibilities:**
- Expose voice tools to Claude agents
- Proxy requests to backend API
- Handle MCP protocol communication

**Tools:**

1. **transcribe_audio**
   ```json
   Input: {"audio_data": "base64_pcm_float32"}
   Output: {"text": "transcription"}
   ```

2. **synthesize_speech**
   ```json
   Input: {"text": "string"}
   Output: {"audio": "base64_wav"}
   ```

3. **voice_conversation**
   ```json
   Input: {
     "user_audio": "base64_audio",
     "agent_response": "text"
   }
   Output: {
     "user_text": "transcription",
     "audio": "base64_wav"
   }
   ```

4. **check_voice_api_health**
   ```json
   Input: {}
   Output: {
     "status": "healthy",
     "whisper_ready": true,
     "orca_ready": true
   }
   ```

**Communication:**
- Outbound: HTTP to backend (internal network)
- Inbound: stdio from Docker exec

**Access:**
```bash
docker compose exec -T mcp-server python server.py
```

## Data Flow Diagrams

### Speech-to-Text Flow

```
┌─────────┐
│ Browser │
│         │
│ 🎤 Mic  │
└────┬────┘
     │ PCM float32, 16kHz
     │ Base64 encoded
     ▼
┌────────────────┐
│   WebSocket    │
│   Connection   │
└────┬───────────┘
     │ {"type": "audio", "data": "..."}
     ▼
┌────────────────┐
│    Backend     │
│                │
│  Audio Buffer  │◄─── 3 second chunks
│  (50% overlap) │
└────┬───────────┘
     │ NumPy array
     ▼
┌────────────────┐
│ Whisper Model  │
│  Transcription │
└────┬───────────┘
     │ Segments
     ▼
┌────────────────┐
│   Process      │
│   Segments     │
│  (dedup, etc)  │
└────┬───────────┘
     │ {"type": "transcription", "text": "..."}
     ▼
┌────────────────┐
│   WebSocket    │
│    Response    │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│    Browser     │
│    Display     │
└────────────────┘
```

### Text-to-Speech Flow

```
┌─────────┐
│ Browser │
│         │
│ 📝 Text │
└────┬────┘
     │ HTTP POST /api/tts
     │ {"text": "Hello"}
     ▼
┌────────────────┐
│    Backend     │
│                │
│  Parse text    │
│  into          │
│  sentences     │
└────┬───────────┘
     │ ["Hello.", "World."]
     ▼
┌────────────────┐
│  Orca Stream   │
│  Synthesizer   │
└────┬───────────┘
     │ PCM int16 chunks
     ▼
┌────────────────┐
│   WAV File     │
│   Builder      │
│  (in memory)   │
└────┬───────────┘
     │ Complete WAV
     ▼
┌────────────────┐
│   Streaming    │
│   Response     │
│  audio/wav     │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│    Browser     │
│  <audio> tag   │
│    Playback    │
└────────────────┘
```

### MCP Tool Call Flow

```
┌──────────────┐
│   Claude     │
│   Desktop    │
└──────┬───────┘
       │ Tool call: transcribe_audio
       │ {"audio_data": "..."}
       ▼
┌──────────────┐
│  MCP Server  │
│              │
│  Parse args  │
│  Validate    │
└──────┬───────┘
       │ HTTP POST /api/stt
       │ {"audio_data": "..."}
       ▼
┌──────────────┐
│   Backend    │
│   API        │
│              │
│  Whisper     │
│  Transcribe  │
└──────┬───────┘
       │ {"text": "..."}
       ▼
┌──────────────┐
│  MCP Server  │
│              │
│  Format      │
│  Response    │
└──────┬───────┘
       │ MCP response
       │ {"text": "User said: ..."}
       ▼
┌──────────────┐
│   Claude     │
│   Desktop    │
│              │
│  Process     │
│  Continue    │
│  Workflow    │
└──────────────┘
```

## Network Architecture

```
┌────────────────────────────────────────────────────┐
│               Host Machine                         │
├────────────────────────────────────────────────────┤
│                                                    │
│  Port Mappings:                                    │
│  • 3000 → frontend:80                             │
│  • 8000 → backend:8000                            │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │     Docker Bridge Network                    │ │
│  │     voice-interaction-network                │ │
│  │     Subnet: 172.x.x.0/16                     │ │
│  ├──────────────────────────────────────────────┤ │
│  │                                              │ │
│  │  Container IPs:                              │ │
│  │  • frontend    → 172.x.x.2                  │ │
│  │  • backend     → 172.x.x.3                  │ │
│  │  • mcp-server  → 172.x.x.4                  │ │
│  │                                              │ │
│  │  Internal DNS:                               │ │
│  │  • frontend    resolves to 172.x.x.2        │ │
│  │  • backend     resolves to 172.x.x.3        │ │
│  │  • mcp-server  resolves to 172.x.x.4        │ │
│  │                                              │ │
│  │  Communication:                              │ │
│  │  frontend → backend:8000                     │ │
│  │  mcp-server → backend:8000                   │ │
│  │                                              │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  Volumes:                                          │
│  • ./models → backend:/app/models (ro)            │
│                                                    │
└────────────────────────────────────────────────────┘
```

## Security Model

### Current Implementation

```
┌─────────────────────────────────────────┐
│           Security Layers               │
├─────────────────────────────────────────┤
│                                         │
│  Layer 1: Network Isolation             │
│  • Docker bridge network                │
│  • Only exposed ports accessible        │
│  • MCP server not exposed               │
│                                         │
│  Layer 2: Container Isolation           │
│  • Separate containers                  │
│  • Minimal base images                  │
│  • Read-only volumes where possible     │
│                                         │
│  Layer 3: Application                   │
│  ⚠️  No authentication                  │
│  ⚠️  Open CORS policy                   │
│  ⚠️  No rate limiting                   │
│  ⚠️  HTTP only (no TLS)                 │
│                                         │
└─────────────────────────────────────────┘
```

### Production Recommendations

```
┌─────────────────────────────────────────┐
│      Enhanced Security Layers           │
├─────────────────────────────────────────┤
│                                         │
│  Layer 1: Network                       │
│  ✅ Reverse proxy (nginx/Traefik)      │
│  ✅ TLS/HTTPS termination              │
│  ✅ Firewall rules                     │
│  ✅ VPC isolation                      │
│                                         │
│  Layer 2: Application                   │
│  ✅ API key authentication             │
│  ✅ Rate limiting (per IP/key)         │
│  ✅ Request validation                 │
│  ✅ CORS whitelist                     │
│                                         │
│  Layer 3: Data                          │
│  ✅ Secrets management (Vault)         │
│  ✅ Encrypted volumes                  │
│  ✅ Audit logging                      │
│  ✅ Data encryption at rest            │
│                                         │
│  Layer 4: Monitoring                    │
│  ✅ Health checks                      │
│  ✅ Metrics (Prometheus)               │
│  ✅ Logging (ELK/Loki)                 │
│  ✅ Alerting                           │
│                                         │
└─────────────────────────────────────────┘
```

## Scaling Strategy

### Horizontal Scaling

```
┌────────────────────────────────────────────────────┐
│              Load Balancer                         │
│              (nginx/HAProxy)                       │
└───┬────────────────┬────────────────┬──────────────┘
    │                │                │
    ▼                ▼                ▼
┌────────┐      ┌────────┐      ┌────────┐
│Backend │      │Backend │      │Backend │
│   #1   │      │   #2   │      │   #3   │
└────────┘      └────────┘      └────────┘
    │                │                │
    └────────────────┴────────────────┘
                     │
                     ▼
         ┌─────────────────────┐
         │  Shared Model Vol   │
         │  (NFS/EBS)          │
         └─────────────────────┘
```

### Service Limits

**Current (Single Instance):**
- Concurrent WebSocket connections: ~100
- Requests per second: ~10-20
- Transcription queue: ~5 concurrent
- TTS queue: ~10 concurrent

**Scaled (3 Instances):**
- Concurrent WebSocket connections: ~300
- Requests per second: ~30-60
- Transcription queue: ~15 concurrent
- TTS queue: ~30 concurrent

## Deployment Workflows

### Development
```
1. Code changes
2. ./deploy.sh stop
3. ./deploy.sh build
4. ./deploy.sh start
5. Test at localhost
```

### CI/CD
```
1. Git push
2. GitHub Actions
   ├─ Run tests
   ├─ Build images
   ├─ Push to registry
   └─ Deploy to staging
3. Manual approval
4. Deploy to production
```

### Production
```
1. docker-compose.prod.yml
2. Use production images
3. Environment-specific config
4. Rolling updates
5. Health checks
6. Rollback capability
```

## Monitoring and Observability

### Metrics to Track

**System Metrics:**
- CPU usage per container
- Memory usage per container
- Network I/O
- Disk usage

**Application Metrics:**
- Request rate (req/sec)
- Response time (p50, p95, p99)
- Error rate (4xx, 5xx)
- WebSocket connections

**Business Metrics:**
- Transcriptions per hour
- Synthesis requests per hour
- Average audio length
- User sessions

### Log Aggregation

```
Container Logs
    ├─ Frontend (nginx access logs)
    ├─ Backend (uvicorn logs)
    └─ MCP Server (stdio logs)
         │
         ▼
    Log Collector
    (Promtail/Fluentd)
         │
         ▼
    Log Storage
    (Loki/Elasticsearch)
         │
         ▼
    Visualization
    (Grafana/Kibana)
```

## Technology Decisions

### Why FastAPI?
- ✅ Native async support
- ✅ Automatic OpenAPI docs
- ✅ Type safety with Pydantic
- ✅ WebSocket support
- ✅ SSE streaming
- ✅ High performance

### Why nginx for Frontend?
- ✅ Production-grade
- ✅ Minimal footprint
- ✅ Excellent caching
- ✅ Compression built-in
- ✅ SPA routing support

### Why Docker Compose?
- ✅ Simple orchestration
- ✅ Perfect for local/staging
- ✅ Easy to understand
- ✅ Quick to deploy
- ✅ Can migrate to Kubernetes later

### Why MCP Protocol?
- ✅ Standard for AI agents
- ✅ Supported by Claude
- ✅ Tool-based interface
- ✅ Extensible
- ✅ Future-proof

## Performance Optimization

### Backend Optimizations
- Model loaded once at startup
- Connection pooling
- Async I/O throughout
- Efficient audio buffering
- Streaming responses

### Frontend Optimizations
- Production build minification
- Gzip compression
- Static asset caching
- Lazy loading
- Code splitting

### Network Optimizations
- WebSocket for real-time
- SSE for streaming
- Binary data as base64
- Chunked transfers
- Keep-alive connections

---

**This architecture enables voice-driven AI interactions at scale.**
