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

# Function to setup ACR
setup_acr() {
    log "Setting up ACR..."
    az acr login -n $ACR_NAME || handle_error "Failed to login to ACR"
    
    # Enable admin access
    log "Enabling admin access for ACR..."
    az acr update --name $ACR_NAME --admin-enabled true
}

# Function to pull and push images with correct architecture
push_image() {
    local source_image=$1
    local target_image=$2
    
    log "Processing image: $source_image -> $target_image"
    
    # Pull the image for AMD64
    docker pull --platform linux/amd64 $source_image || handle_error "Failed to pull $source_image"
    
    # Tag the image
    docker tag $source_image $target_image || handle_error "Failed to tag $source_image"
    
    # Push to ACR
    docker push $target_image || handle_error "Failed to push $target_image"
    
    log "Successfully processed $source_image"
}

# Main process
main() {
    log "Starting image setup process..."
    
    # Setup ACR
    setup_acr
    
    # Process Milvus images
    push_image "milvusdb/milvus:v2.3.3" "$ACR_LOGIN_SERVER/milvus:v2.3.3"
    push_image "quay.io/coreos/etcd:v3.5.5" "$ACR_LOGIN_SERVER/etcd:v3.5.5"
    push_image "minio/minio:RELEASE.2023-03-20T20-16-18Z" "$ACR_LOGIN_SERVER/minio:RELEASE.2023-03-20T20-16-18Z"
    
    log "Image setup completed successfully!"
}

# Execute main process
main