# GitHub Actions Workflows for Experimental Voice Interaction

This directory contains GitHub Actions workflows for building and deploying the experimental voice interaction containers to the self-hosted runner `pop-os-1`.

## Workflows

### 1. Build Base Images (`build-base-images.yml`)

Builds and publishes base Docker images to GitHub Container Registry for faster subsequent builds.

**Triggers:**
- Manual trigger via `workflow_dispatch` with optional force rebuild
- Push to `feature/web` or `main` branches when changes are made to:
  - `experimental/backend/Dockerfile*`
  - `experimental/mcp-server/Dockerfile*`
  - `experimental/web/speech-demo/Dockerfile*`
  - Dependency files (`requirements.txt`, `package*.json`)
- Weekly schedule (Sundays at 2 AM UTC) for security updates

**Services Built:**
- `backend-base` - Python backend API base image
- `mcp-server-base` - MCP (Model Context Protocol) server base image
- `frontend-base` - React frontend base image

**Images Published to:**
- `ghcr.io/<org>/voice-interaction-backend-base:latest`
- `ghcr.io/<org>/voice-interaction-mcp-server-base:latest`
- `ghcr.io/<org>/voice-interaction-frontend-base:latest`

**Features:**
- Smart rebuild detection (only rebuilds when necessary)
- Docker layer caching for faster builds
- Automatic cleanup of old images
- Multi-service matrix build

### 2. Deploy Experimental (`deploy-experimental.yml`)

Deploys the experimental voice interaction stack to the `pop-os-1` self-hosted runner.

**Triggers:**
- Push to `feature/web` or `main` branches with changes to:
  - `experimental/**` directory
  - The workflow file itself
- Manual trigger via `workflow_dispatch` with options:
  - Environment selection (production/staging/development)
  - Force rebuild without cache
  - Cleanup existing containers first

**Services Deployed:**
- Backend API (port 14300)
- Frontend Web App (port 14301)
- MCP Server (stdio transport)
- MCP Server HTTP (port 14302)

**Deployment Process:**
1. Checkout code
2. Set deployment variables based on branch/input
3. Verify system requirements
4. Check/create `.env` file
5. Cleanup existing containers (optional)
6. Pull latest base images
7. Build Docker images with docker-compose
8. Start services
9. Verify deployment health
10. Display service logs and summary

**Health Checks:**
- Backend API health endpoint
- Frontend availability
- MCP Server HTTP health endpoint
- 60-second timeout for all services to become healthy

**Service URLs:**
- Backend API: http://localhost:14300
- Frontend: http://localhost:14301
- MCP Server HTTP: http://localhost:14302

## Prerequisites

### Self-Hosted Runner Setup

The workflows require a self-hosted runner with label `pop-os-1` that has:
- Docker and Docker Compose installed
- GitHub Actions runner configured
- Network access to GitHub Container Registry
- Sufficient disk space for Docker images

### Required Secrets

No additional secrets are required beyond `GITHUB_TOKEN` (automatically provided).

### Environment Variables

The deployment expects a `.env` file in the `experimental/` directory. If not present, it will be created from `.env.example`. Required variables:
- `PICOVOICE_ACCESS_KEY` - API key for Picovoice speech services

## Usage

### Manual Deployment

1. Navigate to Actions tab in GitHub
2. Select "Deploy Experimental to pop-os-1"
3. Click "Run workflow"
4. Choose options:
   - Environment (development/staging/production)
   - Force rebuild (if needed)
   - Cleanup first (recommended)

### Automatic Deployment

Deployments trigger automatically on push to `feature/web` or `main` branches when changes are detected in the `experimental/` directory.

### Building Base Images

Base images are built automatically on relevant changes, but can be manually triggered:

1. Navigate to Actions tab
2. Select "Build Experimental Base Docker Images"
3. Click "Run workflow"
4. Optionally enable "Force rebuild all base images"

## Monitoring

### Check Deployment Status

After deployment, check the workflow run logs for:
- Container status
- Health check results
- Service URLs
- Recent logs

### On the Server

```bash
# View running containers
docker compose -f experimental/docker-compose.yml ps

# View logs
docker compose -f experimental/docker-compose.yml logs -f

# Restart services
docker compose -f experimental/docker-compose.yml restart

# Stop services
docker compose -f experimental/docker-compose.yml down
```

## Troubleshooting

### Deployment Failed

1. Check the workflow logs for specific errors
2. Verify the runner has sufficient resources (disk space, memory)
3. Check service logs on the server
4. Ensure `.env` file has correct values

### Services Not Healthy

If health checks fail:
1. Check individual service logs: `docker compose logs <service-name>`
2. Verify ports are not already in use
3. Check resource availability
4. Review container status: `docker compose ps`

### Base Image Build Failed

1. Verify Dockerfile.base files exist for each service
2. Check GitHub Container Registry permissions
3. Review build logs for specific errors
4. Try force rebuild option

## Notes

- Base images use Docker BuildKit for improved caching
- Deployment includes automatic cleanup of old images (24 hours)
- Health checks wait up to 60 seconds for services to start
- Failed deployments collect diagnostic information automatically
