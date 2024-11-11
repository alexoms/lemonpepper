#!/bin/bash
set -e

# Load environment variables
source .env

# Function to handle errors
handle_error() {
    echo "Error occurred: $1"
    exit 1
}

# Function to log messages
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to process template file
process_template() {
    local template_file=$1
    local output_file=$2
    
    log "Processing template: $template_file"
    
    # Create temp file
    envsubst < $template_file > $output_file
    
    log "Template processed: $output_file"
}

# Function to setup ACR
setup_acr() {
    log "Setting up ACR..."
    
    # Enable admin access
    log "Enabling admin access for ACR..."
    az acr update \
        --name $ACR_NAME \
        --admin-enabled true \
        || handle_error "Failed to enable ACR admin access"
}

# Function to get ACR credentials
get_acr_credentials() {
    log "Getting ACR credentials..."
    
    # Ensure admin access is enabled
    setup_acr
    
    # Get ACR username (usually the ACR name)
    export ACR_USERNAME=$ACR_NAME
    
    # Get ACR password
    export ACR_PASSWORD=$(az acr credential show \
        --name $ACR_NAME \
        --query "passwords[0].value" \
        --output tsv)
        
    if [ -z "$ACR_PASSWORD" ]; then
        handle_error "Failed to get ACR password"
    fi
    
    log "ACR credentials retrieved successfully"
}

# Function to register required resource providers
register_providers() {
    log "Registering required resource providers..."
    
    providers=(
        "Microsoft.App"
        "Microsoft.OperationalInsights"
        "Microsoft.ContainerService"
        "Microsoft.ContainerRegistry"
        "Microsoft.Storage"
    )
    
    for provider in "${providers[@]}"; do
        log "Registering provider: $provider"
        az provider register --namespace $provider --wait
    done
}

# Function to create Log Analytics workspace and get credentials
create_log_analytics() {
    local workspace_name="${CONTAINER_APPS_ENV_NAME}-logs"
    
    log "Creating Log Analytics workspace: $workspace_name"
    
    # Create workspace
    az monitor log-analytics workspace create \
        --resource-group $AZURE_RESOURCE_GROUP \
        --workspace-name $workspace_name \
        --location $AZURE_REGION \
        || handle_error "Failed to create Log Analytics workspace"
    
    # Get workspace ID and key
    export WORKSPACE_ID=$(az monitor log-analytics workspace show \
        --resource-group $AZURE_RESOURCE_GROUP \
        --workspace-name $workspace_name \
        --query customerId \
        --output tsv)
        
    export WORKSPACE_KEY=$(az monitor log-analytics workspace get-shared-keys \
        --resource-group $AZURE_RESOURCE_GROUP \
        --workspace-name $workspace_name \
        --query primarySharedKey \
        --output tsv)
    
    log "Log Analytics workspace created successfully"
}

# Create Container Apps Environment if it doesn't exist
create_environment() {
    log "Creating Container Apps Environment..."
    
    # First create Log Analytics workspace and get credentials
    create_log_analytics
    
    log "Creating Container Apps Environment with Log Analytics integration..."
    az containerapp env create \
        --name $CONTAINER_APPS_ENV_NAME \
        --resource-group $AZURE_RESOURCE_GROUP \
        --location $AZURE_REGION \
        --logs-workspace-id $WORKSPACE_ID \
        --logs-workspace-key $WORKSPACE_KEY \
        || handle_error "Failed to create Container Apps Environment"
}

# Deploy Milvus service
deploy_milvus_service() {
    log "Deploying Milvus service..."
    
    # Create storage account if it doesn't exist
    az storage account create \
        --name $STORAGE_ACCOUNT_NAME \
        --resource-group $AZURE_RESOURCE_GROUP \
        --location $AZURE_REGION \
        --sku Standard_LRS \
        || log "Storage account already exists"

    # Get storage key
    STORAGE_KEY=$(az storage account keys list \
        --resource-group $AZURE_RESOURCE_GROUP \
        --account-name $STORAGE_ACCOUNT_NAME \
        --query "[0].value" -o tsv)
    export STORAGE_KEY

    # Create file shares if they don't exist
    for share in $STORAGE_SHARE_ETCD $STORAGE_SHARE_MINIO; do
        az storage share create \
            --name $share \
            --account-name $STORAGE_ACCOUNT_NAME \
            --account-key $STORAGE_KEY \
            || log "Share $share already exists"
    done

    # Get ACR credentials
    export ACR_USERNAME=$ACR_NAME
    export ACR_PASSWORD=$(az acr credential show \
        --name $ACR_NAME \
        --query "passwords[0].value" \
        --output tsv)

    # Create environment variables for storage config
    #export ETCD_STORAGE_NAME=$STORAGE_SHARE_ETCD
    #export MINIO_STORAGE_NAME=$STORAGE_SHARE_MINIO
    #export STORAGE_ACCOUNT_NAME_VALUE=$STORAGE_ACCOUNT_NAME

    # Explicitly export all variables needed for envsubst
    export STORAGE_KEY
    export STORAGE_SHARE_ETCD
    export STORAGE_SHARE_MINIO
    export STORAGE_ACCOUNT_NAME
    export AZURE_LOCATION=$AZURE_REGION
    export AZURE_SUBSCRIPTION_ID
    export AZURE_RESOURCE_GROUP
    export CONTAINER_APPS_ENV_NAME
    export ACR_LOGIN_SERVER
    export AZURE_REGION

    # Process template
    local template_file="milvus-deployment.yaml"
    local processed_file="milvus-deployment-processed.yaml"
    
    log "Processing template: $template_file"
    envsubst '$AZURE_SUBSCRIPTION_ID $AZURE_RESOURCE_GROUP $CONTAINER_APPS_ENV_NAME $ACR_LOGIN_SERVER $ACR_USERNAME $ACR_PASSWORD $STORAGE_KEY $STORAGE_SHARE_ETCD $STORAGE_SHARE_MINIO $STORAGE_ACCOUNT_NAME $AZURE_REGION' < $template_file > $processed_file
    log "Template processed: $processed_file"

    # Show processed file for debugging
    log "Processed YAML content:"
    cat $processed_file

    log "Mounting storage:"
    az containerapp env storage set --access-mode ReadWrite --azure-file-account-name $STORAGE_ACCOUNT_NAME --azure-file-account-key $STORAGE_KEY --azure-file-share-name $STORAGE_SHARE_ETCD --storage-name $STORAGE_SHARE_ETCD --name $CONTAINER_APPS_ENV_NAME --resource-group $AZURE_RESOURCE_GROUP --output table
    az containerapp env storage set --access-mode ReadWrite --azure-file-account-name $STORAGE_ACCOUNT_NAME --azure-file-account-key $STORAGE_KEY --azure-file-share-name $STORAGE_SHARE_MINIO --storage-name $STORAGE_SHARE_MINIO --name $CONTAINER_APPS_ENV_NAME --resource-group $AZURE_RESOURCE_GROUP --output table

    # Deploy Milvus using the processed template
    log "Creating Milvus container app..."
    az containerapp update \
        --name milvus-service \
        --resource-group $AZURE_RESOURCE_GROUP \
        --yaml $processed_file \
        || az containerapp create \
            --name milvus-service \
            --resource-group $AZURE_RESOURCE_GROUP \
            --yaml $processed_file \
        || handle_error "Failed to deploy Milvus service"
        
    # Clean up processed file
    #rm $processed_file
}

# Deploy Ollama service
deploy_ollama_service() {
    log "Deploying Ollama service..."
    az containerapp create \
        --name ollama-service \
        --resource-group $AZURE_RESOURCE_GROUP \
        --environment $CONTAINER_APPS_ENV_NAME \
        --image $ACR_LOGIN_SERVER/ollama:latest \
        --target-port 11434 \
        --ingress internal \
        --registry-server $ACR_LOGIN_SERVER \
        --registry-username $ACR_USERNAME \
        --registry-password $ACR_PASSWORD \
        --cpu 4 \
        --memory 8Gi \
        --min-replicas 1 \
        --max-replicas 1 \
        || handle_error "Failed to deploy Ollama service"
}

# Deploy RAG service
deploy_rag_service() {
    log "Deploying RAG service..."
    az containerapp create \
        --name rag-service \
        --resource-group $AZURE_RESOURCE_GROUP \
        --environment $CONTAINER_APPS_ENV_NAME \
        --image $ACR_LOGIN_SERVER/rag-service:latest \
        --target-port 8000 \
        --ingress external \
        --registry-server $ACR_LOGIN_SERVER \
        --registry-username $ACR_USERNAME \
        --registry-password $ACR_PASSWORD \
        --cpu 1 \
        --memory 2Gi \
        --env-vars "MILVUS_HOST=milvus-service" "MILVUS_PORT=19530" "OLLAMA_SERVER=http://ollama-service:11434" \
        || handle_error "Failed to deploy RAG service"
}

# Main deployment process
main() {
    log "Starting deployment process..."
    
    # Ensure ACR login and get credentials
    az acr login -n $ACR_NAME || handle_error "Failed to login to ACR"
    get_acr_credentials
    
    # Register required providers
    register_providers
    
    # Create environment
    create_environment
    
    # Deploy services
    deploy_milvus_service
    deploy_ollama_service
    deploy_rag_service
    
    log "Deployment completed successfully!"
    
    # Display the RAG service URL
    local rag_url=$(az containerapp show \
        --name rag-service \
        --resource-group $AZURE_RESOURCE_GROUP \
        --query "properties.configuration.ingress.fqdn" \
        --output tsv)
    
    log "RAG Service is available at: https://$rag_url"
}

# Execute deployment
main