#!/bin/bash
set -e

# Load environment variables
source .env

# Function to replace variables in YAML files
replace_variables() {
    local input_file=$1
    local output_file=$2
    
    cat $input_file | \
    sed "s|your-acr.azurecr.io|$ACR_LOGIN_SERVER|g" | \
    sed "s|your-resource-group|$AZURE_RESOURCE_GROUP|g" | \
    sed "s|{subscription-id}|$AZURE_SUBSCRIPTION_ID|g" | \
    sed "s|{environment-name}|$CONTAINER_APPS_ENV_NAME|g" > $output_file
}

# Create temporary deployment files with replaced variables
echo "Preparing deployment files..."
replace_variables rag-service.yaml rag-service-deploy.yaml
replace_variables ollama-service.yaml ollama-service-deploy.yaml
replace_variables milvus-service.yaml milvus-service-deploy.yaml

# Deploy services
echo "Deploying services..."
az containerapp create \
    -n rag-service \
    --resource-group $AZURE_RESOURCE_GROUP \
    --yaml rag-service-deploy.yaml

az containerapp create \
    -n ollama-service \
    --resource-group $AZURE_RESOURCE_GROUP \
    --yaml ollama-service-deploy.yaml

az containerapp create \
    -n milvus-service \
    --resource-group $AZURE_RESOURCE_GROUP \
    --yaml milvus-service-deploy.yaml

# Clean up temporary files
rm rag-service-deploy.yaml ollama-service-deploy.yaml milvus-service-deploy.yaml

echo "Deployment completed successfully!"