#!/bin/bash
set -e

# Load environment variables
source .env

# Create Azure Container Registry if it doesn't exist
echo "Creating Azure Container Registry..."
az acr create \
    --resource-group $AZURE_RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic

# Login to ACR
echo "Logging into ACR..."
az acr login --name $ACR_NAME

# Create storage account if it doesn't exist
echo "Creating Storage Account..."
az storage account create \
    --name $STORAGE_ACCOUNT_NAME \
    --resource-group $AZURE_RESOURCE_GROUP \
    --location $AZURE_REGION \
    --sku Standard_LRS

# Get storage account key
STORAGE_KEY=$(az storage account keys list \
    --resource-group $AZURE_RESOURCE_GROUP \
    --account-name $STORAGE_ACCOUNT_NAME \
    --query "[0].value" -o tsv)

# Create file shares
echo "Creating Azure File Shares..."
az storage share create \
    --name $STORAGE_SHARE_OLLAMA \
    --account-name $STORAGE_ACCOUNT_NAME \
    --account-key "$STORAGE_KEY"

az storage share create \
    --name $STORAGE_SHARE_ETCD \
    --account-name $STORAGE_ACCOUNT_NAME \
    --account-key "$STORAGE_KEY"

az storage share create \
    --name $STORAGE_SHARE_MINIO \
    --account-name $STORAGE_ACCOUNT_NAME \
    --account-key "$STORAGE_KEY"

echo "Setup completed successfully!"