# LLM-Powered Real-Time Audio Transcription and Analysis

## Overview

This application provides real-time audio transcription and analysis using Large Language Models (LLMs). It captures audio input, transcribes it, and processes the transcription through an LLM to generate intelligent responses. The system is designed for various use cases, including interview assistance, coding help, and general question-answering.

## Features

- Real-time audio transcription using either Vosk or Google Cloud Speech-to-Text
- Integration with Ollama API for LLM-powered responses
- Customizable prompts for different types of interactions (e.g., coding, non-coding interviews)
- Dynamic audio visualization with level monitoring
- Session management for keeping track of multiple conversations
- User-friendly GUI built with Textual
- Adjustable audio settings including device selection and gain control
- Copy-to-clipboard functionality for easy sharing of transcriptions and LLM responses

## Prerequisites

- Python 3.7+
- Ollama (for local LLM integration)
- Vosk (for offline speech recognition)
- Google Cloud SDK (for Google Cloud Speech-to-Text, optional)

![Home](docs/images/screenshot_home.png)

[Watch the demo video](docs/video/demo-realtimeaudioai.mp4)

## Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up Ollama and ensure it's running on the specified host (default: http://localhost:11434)

4. If using Google Cloud Speech-to-Text, set up your Google Cloud credentials

## Usage

Run the application:

```
python gui_textual.py
```

1. Select an audio input device from the available options.
2. Choose the transcription method (Vosk or Google Cloud).
3. Select a prompt template based on your use case.
4. Start speaking, and the application will transcribe your audio in real-time.
5. The LLM will process the transcription and provide responses.
6. Use the various buttons to control the application, including pausing/resuming transcription, clearing data, and copying responses.

## Key Components

- `gui_textual.py`: Main application GUI and control logic
- `ollama_api.py`: Integration with Ollama for LLM processing
- `transcribe_audio.py`: Vosk-based audio transcription
- `transcribe_audio_google_cloud.py`: Google Cloud Speech-to-Text integration

## Customization

- Modify prompts in `ollama_api.py` to tailor the LLM's responses to your needs
- Adjust audio processing parameters in `transcribe_audio.py` for optimal performance
- Customize the GUI layout and styling in `gui_textual.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

TBD

## Acknowledgements

- Vosk for offline speech recognition
- Google Cloud Speech-to-Text for online transcription
- Ollama for local LLM integration
- Textual for the TUI framework

---
