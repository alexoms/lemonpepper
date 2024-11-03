# Use CUDA-enabled base image
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

# Install Python and required system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY rag_microservice.py .
COPY langchain_milvus_rag_chat_api.py .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python3", "rag_microservice.py"]