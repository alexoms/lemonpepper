# Use multi-stage build for better dependency management
FROM --platform=linux/amd64 nvidia/cuda:12.0.0-runtime-ubuntu22.04 as builder

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and required system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create Python symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set up virtual environment
RUN python3 -m pip install --no-cache-dir virtualenv
RUN python3 -m virtualenv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies in order
WORKDIR /app
COPY requirements.txt .

# Install core dependencies first
RUN pip install --no-cache-dir \
    langchain-core>=0.2.38 \
    pydantic==2.4.2 \
    fastapi==0.104.1 \
    uvicorn==0.24.0

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM --platform=linux/amd64 nvidia/cuda:12.0.0-runtime-ubuntu22.04

# Install Python in the final stage
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create Python symlinks in final stage
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy Python and dependencies from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/lib /usr/lib

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/venv/lib/python3.10/site-packages:$PYTHONPATH"

# Set working directory
WORKDIR /app

# Copy application files
COPY rag_microservice.py .
COPY langchain_milvus_rag_chat_api.py .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python3", "rag_microservice.py"]