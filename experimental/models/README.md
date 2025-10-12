# Whisper Models Directory

This directory should contain Whisper model files for speech recognition.

## Downloading Models

### Recommended Model (Base English)

```bash
# Download base English model (~150MB)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

### Available Models

| Model | Size | Description | URL |
|-------|------|-------------|-----|
| tiny.en | 75MB | Fastest, lowest accuracy | [Download](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin) |
| base.en | 150MB | Good balance | [Download](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin) |
| small.en | 500MB | Better accuracy | [Download](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin) |
| medium.en | 1.5GB | Best accuracy | [Download](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin) |

### Multilingual Models

For non-English languages, use multilingual models:

```bash
# Base multilingual model
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
```

## Usage

After downloading, update your `.env` file:

```bash
WHISPER_MODEL_PATH=/app/models/ggml-base.en.bin
```

## Quick Setup Script

```bash
#!/bin/bash
# download_model.sh

cd "$(dirname "$0")"

MODEL_NAME=${1:-ggml-base.en.bin}
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$MODEL_NAME"

echo "Downloading $MODEL_NAME..."
wget -O "$MODEL_NAME" "$MODEL_URL"

if [ $? -eq 0 ]; then
    echo "✓ Downloaded successfully: $MODEL_NAME"
    echo "  Size: $(du -h "$MODEL_NAME" | cut -f1)"
else
    echo "✗ Download failed"
    exit 1
fi
```

Usage:
```bash
chmod +x download_model.sh
./download_model.sh ggml-base.en.bin
```

## Notes

- Models are read-only when mounted in Docker
- Larger models provide better accuracy but slower processing
- English-only models (`.en`) are faster for English speech
- Models are cached on first load for faster subsequent starts
