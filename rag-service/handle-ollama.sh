#!/bin/bash
set -e

# Load environment variables
source .env

# Function to handle errors
handle_error() {
    echo "Error occurred: $1"
    exit 1
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

# Main execution
echo "Ensuring ACR login..."
az acr login -n $ACR_NAME || handle_error "ACR login failed"

echo "Pulling Ollama image..."
retry_command "docker pull ollama/ollama:latest"

echo "Tagging Ollama image..."
docker tag ollama/ollama:latest $ACR_LOGIN_SERVER/ollama:latest

echo "Pushing Ollama image..."
retry_command "docker push $ACR_LOGIN_SERVER/ollama:latest"

echo "Ollama image processing completed successfully!"