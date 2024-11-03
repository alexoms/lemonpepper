# Deploy services
az containerapp create -n rag-service --resource-group your-resource-group --yaml rag-service.yaml
az containerapp create -n ollama-service --resource-group your-resource-group --yaml ollama-service.yaml
az containerapp create -n milvus-service --resource-group your-resource-group --yaml milvus-service.yaml
