# Create storage account
az storage account create \
    --name yourstorageaccount \
    --resource-group your-resource-group \
    --location eastus \
    --sku Standard_LRS

# Create file shares
az storage share create \
    --name ollama-storage \
    --account-name yourstorageaccount

az storage share create \
    --name etcd-storage \
    --account-name yourstorageaccount

az storage share create \
    --name minio-storage \
    --account-name yourstorageaccount