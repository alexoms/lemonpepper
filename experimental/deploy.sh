#!/bin/bash
# Voice Interaction API - Deployment Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    log_info "✓ Docker and Docker Compose are installed"
}

# Check environment file
check_environment() {
    if [ ! -f ".env" ]; then
        log_warn ".env file not found"
        if [ -f ".env.example" ]; then
            log_info "Copying .env.example to .env"
            cp .env.example .env
            log_warn "Please edit .env and add your PICOVOICE_ACCESS_KEY"
            log_info "Get your key from: https://console.picovoice.ai/"
            exit 1
        else
            log_error ".env.example not found"
            exit 1
        fi
    fi

    # Check if Picovoice key is set
    if grep -q "your_picovoice_access_key" .env; then
        log_error "Please set PICOVOICE_ACCESS_KEY in .env file"
        exit 1
    fi

    log_info "✓ Environment configured"
}

# Check Whisper model
check_model() {
    log_info "Checking for Whisper model..."

    if [ ! -d "models" ]; then
        mkdir -p models
    fi

    MODEL_FILE="models/ggml-base.en.bin"

    if [ ! -f "$MODEL_FILE" ]; then
        log_warn "Whisper model not found"
        log_info "Downloading base English model (~150MB)..."

        cd models
        wget -q --show-progress \
            https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin \
            || {
                log_error "Failed to download model"
                exit 1
            }
        cd ..

        log_info "✓ Model downloaded successfully"
    else
        log_info "✓ Whisper model found"
    fi
}

# Build services
build_services() {
    log_info "Building Docker services..."
    docker-compose build || {
        log_error "Failed to build services"
        exit 1
    }
    log_info "✓ Services built successfully"
}

# Start services
start_services() {
    log_info "Starting services..."
    docker-compose up -d || {
        log_error "Failed to start services"
        exit 1
    }

    log_info "Waiting for services to be ready..."
    sleep 5

    # Check health
    if curl -sf http://localhost:14300/health > /dev/null; then
        log_info "✓ Backend is healthy"
    else
        log_warn "Backend may not be ready yet"
    fi

    log_info "✓ Services started successfully"
}

# Stop services
stop_services() {
    log_info "Stopping services..."
    docker-compose down
    log_info "✓ Services stopped"
}

# Show status
show_status() {
    log_info "Service Status:"
    docker-compose ps

    echo ""
    log_info "Health Check:"
    curl -s http://localhost:14300/health | python -m json.tool || log_warn "Backend not responding"
}

# Show logs
show_logs() {
    SERVICE=${1:-}
    if [ -z "$SERVICE" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$SERVICE"
    fi
}

# Show URLs
show_urls() {
    echo ""
    log_info "==================================="
    log_info "Voice Interaction API is running!"
    log_info "==================================="
    echo ""
    echo "  Frontend:     http://localhost:3000"
    echo "  API Docs:     http://localhost:14300/docs"
    echo "  Health Check: http://localhost:14300/health"
    echo ""
    log_info "To view logs: ./deploy.sh logs [service]"
    log_info "To stop: ./deploy.sh stop"
    echo ""
}

# Main script
main() {
    ACTION=${1:-deploy}

    case "$ACTION" in
        deploy)
            log_info "Starting deployment..."
            check_prerequisites
            check_environment
            check_model
            build_services
            start_services
            show_urls
            ;;

        start)
            start_services
            show_urls
            ;;

        stop)
            stop_services
            ;;

        restart)
            stop_services
            start_services
            show_urls
            ;;

        build)
            build_services
            ;;

        status)
            show_status
            ;;

        logs)
            show_logs "$2"
            ;;

        clean)
            log_warn "This will remove all containers and volumes"
            read -p "Are you sure? (y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                docker-compose down -v
                log_info "✓ Cleaned up"
            fi
            ;;

        help|*)
            echo "Voice Interaction API - Deployment Script"
            echo ""
            echo "Usage: ./deploy.sh [command]"
            echo ""
            echo "Commands:"
            echo "  deploy    - Full deployment (check, build, start)"
            echo "  start     - Start services"
            echo "  stop      - Stop services"
            echo "  restart   - Restart services"
            echo "  build     - Build Docker images"
            echo "  status    - Show service status"
            echo "  logs      - Show logs (optional: specify service)"
            echo "  clean     - Remove containers and volumes"
            echo "  help      - Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./deploy.sh deploy"
            echo "  ./deploy.sh logs backend"
            echo "  ./deploy.sh restart"
            ;;
    esac
}

main "$@"
