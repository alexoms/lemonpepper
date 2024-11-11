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

# Function to check if a resource exists before trying to delete it
resource_exists() {
    local resource_type=$1
    local resource_name=$2
    
    az resource show \
        --resource-group $AZURE_RESOURCE_GROUP \
        --name $resource_name \
        --resource-type "Microsoft.App/containerApps" \
        --query "name" \
        --output tsv 2>/dev/null
}

# Function to wait for deletion
wait_for_deletion() {
    local resource_name=$1
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if ! resource_exists "Microsoft.App/containerApps" $resource_name; then
            return 0
        fi
        log "Waiting for $resource_name to be deleted... (Attempt $attempt/$max_attempts)"
        sleep 10
        attempt=$((attempt + 1))
    done
    
    handle_error "Timeout waiting for $resource_name to be deleted"
}

# Function to delete Azure File Share
delete_file_share() {
    local share_name=$1
    log "Deleting Azure File Share: $share_name"
    
    if az storage share exists --name $share_name --account-name $STORAGE_ACCOUNT_NAME --output tsv 2>/dev/null; then
        az storage share delete \
            --name $share_name \
            --account-name $STORAGE_ACCOUNT_NAME \
            --only-show-errors || log "Warning: Could not delete file share $share_name"
    else
        log "File share $share_name does not exist"
    fi
}

# Main undeployment process
main() {
    log "Starting undeployment process..."

    # Delete Container Apps
    local apps=("rag-service" "ollama-service" "milvus-service")
    
    for app in "${apps[@]}"; do
        if resource_exists "Microsoft.App/containerApps" $app; then
            log "Deleting Container App: $app"
            az containerapp delete \
                --name $app \
                --resource-group $AZURE_RESOURCE_GROUP \
                --yes \
                --only-show-errors || log "Warning: Could not delete $app"
            
            wait_for_deletion $app
        else
            log "Container App $app does not exist"
        fi
    done

    # Delete Azure File Shares
    # local shares=("$STORAGE_SHARE_ETCD" "$STORAGE_SHARE_MINIO" "$STORAGE_SHARE_OLLAMA")
    
    # for share in "${shares[@]}"; do
    #     delete_file_share $share
    # done

    # Clean up ACR repositories
    # log "Cleaning up ACR repositories..."
    # local repositories=("rag-service" "ollama" "milvus" "etcd" "minio")
    
    # for repo in "${repositories[@]}"; do
    #     if az acr repository show \
    #         --name $ACR_NAME \
    #         --repository $repo \
    #         --output none 2>/dev/null; then
            
    #         log "Deleting repository: $repo"
    #         az acr repository delete \
    #             --name $ACR_NAME \
    #             --repository $repo \
    #             --yes \
    #             --only-show-errors || log "Warning: Could not delete repository $repo"
    #     else
    #         log "Repository $repo does not exist in ACR"
    #     fi
    # done

    # Optional: Delete the Container Apps Environment
    # if [ ! -z "$CONTAINER_APPS_ENV_NAME" ]; then
    #     log "Deleting Container Apps Environment: $CONTAINER_APPS_ENV_NAME"
    #     az containerapp env delete \
    #         --name $CONTAINER_APPS_ENV_NAME \
    #         --resource-group $AZURE_RESOURCE_GROUP \
    #         --yes \
    #         --only-show-errors || log "Warning: Could not delete Container Apps Environment"
    # fi

    # Optional: Delete the Storage Account
    # if [ "$DELETE_STORAGE_ACCOUNT" = "true" ]; then
    #     log "Deleting Storage Account: $STORAGE_ACCOUNT_NAME"
    #     az storage account delete \
    #         --name $STORAGE_ACCOUNT_NAME \
    #         --resource-group $AZURE_RESOURCE_GROUP \
    #         --yes \
    #         --only-show-errors || log "Warning: Could not delete Storage Account"
    # fi

    log "Undeployment completed!"
}

# Confirmation prompt
confirm_undeploy() {
    read -p "This will delete all deployed resources. Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Undeployment cancelled"
        exit 1
    fi
}

# Execute with confirmation
confirm_undeploy
main