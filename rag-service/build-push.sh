#!/bin/bash
set -e

# Load environment variables
source .env

# Function to handle errors
handle_error() {
    echo "Error occurred: $1"
    exit 1
}

# Function to check if Docker is running and responsive
check_docker() {
    docker info >/dev/null 2>&1 || {
        echo "Docker is not running or not responding. Please start Docker Desktop and try again."
        exit 1
    }
}

# Function to retry a command
retry_command() {
    local retries=3
    local count=0
    local delay=5
    while [ $count -lt $retries ]; do
        count=$((count + 1))
        echo "Attempt $count of $retries: $1"
        if eval "$1"; then
            return 0
        fi
        echo "Command failed. Waiting $delay seconds before retrying..."
        sleep $delay
    done
    handle_error "Failed after $retries attempts: $1"
}

# Build and push RAG service
build_push_rag() {
    echo "Ensuring ACR login..."
    az acr login -n $ACR_NAME || handle_error "ACR login failed"

    echo "Building RAG service for AMD64..."
    retry_command "docker build --platform linux/amd64 -t $ACR_LOGIN_SERVER/rag-service:latest ."

    echo "Pushing RAG service..."
    retry_command "docker push $ACR_LOGIN_SERVER/rag-service:latest"
}

# Handle Ollama image separately
handle_ollama() {
    echo "Handling Ollama image..."
    
    # Try to pull the image
    if ! retry_command "docker pull --platform linux/amd64 ollama/ollama:latest"; then
        echo "Failed to pull Ollama image directly. Trying alternative approach..."
        return 1
    fi

    echo "Tagging Ollama image..."
    if ! docker tag ollama/ollama:latest $ACR_LOGIN_SERVER/ollama:latest; then
        echo "Failed to tag Ollama image"
        return 1
    fi

    echo "Pushing Ollama image..."
    if ! retry_command "docker push $ACR_LOGIN_SERVER/ollama:latest"; then
        echo "Failed to push Ollama image"
        return 1
    fi

    return 0
}

# Main execution
main() {
    check_docker

    # Build and push RAG service
    build_push_rag

    # Try to handle Ollama image
    if ! handle_ollama; then
        echo "Warning: Failed to process Ollama image. You may need to run the ollama script separately."
        echo "You can try running just the Ollama part later with:"
        echo "./handle-ollama.sh"
    fi
}

# Run main function
main