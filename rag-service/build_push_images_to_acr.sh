# Login to ACR
az acr login --name your-acr

# Build and push RAG service
docker build -t your-acr.azurecr.io/rag-service:latest .
docker push your-acr.azurecr.io/rag-service:latest

# Push Ollama image
docker pull ollama/ollama:latest
docker tag ollama/ollama:latest your-acr.azurecr.io/ollama:latest
docker push your-acr.azurecr.io/ollama:latest