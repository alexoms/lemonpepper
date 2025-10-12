# Port Configuration

## Overview

All services use ports in the **14300 range** to avoid conflicts with commonly used ports.

## Port Assignments

| Service | Internal Port | External Port | URL |
|---------|---------------|---------------|-----|
| **Backend API** | 8000 | **14300** | http://localhost:14300 |
| **Frontend Web** | 80 | **14301** | http://localhost:14301 |
| **MCP Server HTTP** | 8001 | **14302** | http://localhost:14302 |
| **MCP Server stdio** | N/A | N/A | (Internal only) |

## Access URLs

### Web Interface
```
http://localhost:14301
```

### API Documentation (Swagger)
```
http://localhost:14300/docs
```

### API Health Check
```
http://localhost:14300/health
```

### MCP HTTP Server
```
http://localhost:14302/tools
```

## Configuration

### Docker Compose
Ports are configured in `docker-compose.yml`:

```yaml
services:
  backend:
    ports:
      - "14300:8000"  # Backend API

  frontend:
    ports:
      - "14301:80"    # Frontend Web

  mcp-server-http:
    ports:
      - "14302:8001"  # MCP HTTP Server
```

### Environment Variables

**Backend:**
- Container listens on port 8000 internally
- Mapped to host port 14300

**Frontend:**
- Container listens on port 80 internally
- Mapped to host port 14301
- Configured to connect to backend at http://localhost:14300

**MCP Server:**
- Container listens on port 8001 internally
- Mapped to host port 14302
- Environment: `MCP_HTTP_PORT=8001` (internal), `MCP_HTTP_HOST=0.0.0.0`

## Testing Connectivity

```bash
# Test backend
curl http://localhost:14300/health

# Test frontend
curl http://localhost:14301/health

# Test MCP server
curl http://localhost:14302/health

# Check all services
docker-compose ps
```

## Firewall Configuration

If running on a server, open these ports:

```bash
# Ubuntu/Debian with ufw
sudo ufw allow 14300/tcp comment "Voice API Backend"
sudo ufw allow 14301/tcp comment "Voice API Frontend"
sudo ufw allow 14302/tcp comment "Voice API MCP HTTP"

# CentOS/RHEL with firewalld
sudo firewall-cmd --permanent --add-port=14300/tcp
sudo firewall-cmd --permanent --add-port=14301/tcp
sudo firewall-cmd --permanent --add-port=14302/tcp
sudo firewall-cmd --reload
```

## Port Conflicts

If you encounter port conflicts, you can change the external ports in `docker-compose.yml`:

```yaml
ports:
  - "YOUR_PORT:8000"  # Change YOUR_PORT to any available port
```

Remember to also update:
1. `.env` file: `REACT_APP_API_URL=http://localhost:YOUR_PORT`
2. Frontend environment in docker-compose.yml

## Internal Docker Network

Services communicate internally using Docker network names:
- Frontend → Backend: `http://backend:8000` (internal)
- MCP Server → Backend: `http://backend:8000` (internal)

External access uses the mapped ports (14300-14302).

## Development vs Production

### Development (localhost)
- Use ports 14300-14302 on localhost
- No authentication required
- Direct Docker port mapping

### Production
- Use reverse proxy (nginx, Traefik)
- Map to standard ports (80, 443)
- Enable HTTPS/TLS
- Add authentication
- Example nginx config:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:14300;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

server {
    listen 80;
    server_name app.yourdomain.com;

    location / {
        proxy_pass http://localhost:14301;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Quick Reference

```bash
# Start all services
docker-compose up -d

# Check which ports are in use
docker-compose ps

# View logs
docker-compose logs -f

# Test all endpoints
curl http://localhost:14300/health  # Backend
curl http://localhost:14301/health  # Frontend
curl http://localhost:14302/health  # MCP Server
```

## Troubleshooting

### Port Already in Use
```bash
# Check what's using the port
sudo lsof -i :14300
sudo lsof -i :14301
sudo lsof -i :14302

# Kill the process if needed
sudo kill -9 <PID>

# Or change ports in docker-compose.yml
```

### Cannot Connect
```bash
# Verify services are running
docker-compose ps

# Check if ports are exposed
docker-compose port backend 8000
docker-compose port frontend 80
docker-compose port mcp-server-http 8001

# Check firewall
sudo ufw status
```

### Wrong Port in Frontend
```bash
# Rebuild frontend with correct API URL
docker-compose build --no-cache frontend
docker-compose up -d frontend
```

## Summary

- 🔵 **14300** - Backend API (FastAPI)
- 🟢 **14301** - Frontend Web (React + nginx)
- 🟣 **14302** - MCP Server HTTP (FastAPI)

All ports are in the 14300 range to avoid conflicts with standard services.
