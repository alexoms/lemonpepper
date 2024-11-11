#!/bin/bash
set -e

# Load environment variables
source .env
# Check images in ACR
az acr repository list -n $ACR_NAME

# Show tags for RAG service
az acr repository show-tags -n $ACR_NAME --repository rag-service

# Show tags for Ollama
az acr repository show-tags -n $ACR_NAME --repository ollama

# Get details of the latest RAG service image
az acr repository show -n $ACR_NAME --image rag-service:latest

# Get details of the latest Ollama image
az acr repository show -n $ACR_NAME --image ollama:latest

# Optional: Check image vulnerabilities (if you have container scan enabled)
az acr scan show-findings -r $ACR_NAME --repository rag-service:latest