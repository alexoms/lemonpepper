#!/bin/bash
set -e

# Load environment variables
source .env

# Build and push RAG service
echo "Building and pushing RAG service..."
docker build -t $ACR_LOGIN_SERVER/rag-service:latest .
docker push $ACR_LOGIN_SERVER/rag-service:latest

# Push Ollama image
echo "Pushing Ollama image..."
docker pull ollama/ollama:latest
docker tag ollama/ollama:latest $ACR_LOGIN_SERVER/ollama:latest
docker push $ACR_LOGIN_SERVER/ollama:latest

echo "Build and push completed successfully!"