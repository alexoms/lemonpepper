# Co-Pilot AI Agent using Offline Real-Time Audio Transcription and Offline LLM-Powered Analysis

## Overview

This application provides real-time audio transcription and analysis using Large Language Models (LLMs). It captures audio input, transcribes it, and processes the transcription through an LLM to generate intelligent responses. The system is designed for various use cases, including interview assistance, coding help, and general question-answering.  Customizable prompt templates allow for testing multiple audio driven agent use cases such as co-piloting with an audio feed, direct multi-modal human voice interaction, etc.  

## Features

- Real-time audio transcription using either Vosk or Google Cloud Speech-to-Text
- Integration with Ollama API for LLM-powered responses
- Customizable prompts for different types of interactions (e.g., interviews, summaries, helpful assistance, conversation analysis, sentiment analysis, puzzle solving, design architecting, code ideation, business strategy recommender)
- Dynamic audio visualization with level monitoring
- Session management for keeping track of multiple conversations
- User-friendly GUI built with Textual
- Adjustable audio settings including device selection and gain control
- Copy-to-clipboard functionality for easy sharing of transcriptions and LLM responses

## Prerequisites

- Python 3.7+
- Ollama (for local LLM integration)
- Whisper (for offline speech recognition).  
- Optional...Vosk (for offline speech recognition).  Audio language models can be downloaded at https://alphacephei.com/vosk/models
- Optional...Google Cloud SDK (for Google Cloud Speech-to-Text, optional)

- Headset/microphone (supports bluetooth)
- Software audio loopback for mixing microphone and other audio source such as from a browser or video conferencing tool 
  
  Users can pipe audio on their machines using Mac MIDI Studio, Existential Audio's BlackHole, Rogue Amoeba's Loopback, etc.  On windows, Jack Audio Connection Kit, VB Audio, etc.  Ubuntu users might use PulseAudio, etc.

[![Watch the demo video](docs/images/screenshot_home.png)](https://youtu.be/X29zS82GMVk?hd=1 "Demo")

## Installation

> **_OPTIONAL:_** It is recommended to utilize a Python version management system, such as Miniconda for this step:  
> - To create an environment, use: **_conda create -n py310-audio-ai python=3.10 -y_**
> - To activate the environment, use: **_conda activate py310-audio-ai_**

1. Clone the repository:
   ```
   git clone https://github.com/alexoms/lemonpepper.git
   cd lemonpepper
   ```

2. Install required dependencies:
   ```
   pip install lemonpepper

   or 

   pip install .
   ```

3. Download and setup Ollama (https://ollama.com/) with your desired LLM (https://ollama.com/library) and ensure it's running on the specified host (default: http://localhost:11434)


## Usage

Run the application:

```
lemonpepper
```

1. Click on the settings tab
2. Point to the ollama server that you have installed already.  The value of the Ollama API Settings form field is preset to http://localhost:11434.  You can change it to another host by editing the form field and changing the localhost to the ip address if you have install ollama to a different machine.
3. If you change the host name, click on the Refresh Models button under the hostname to pull the list of llm models that this ollama server has available.
4. Click on the Select Ollama Model dropdown and it should show an updated list of available models
5. If you do not see any models, then log into your ollama server and make sure to pull a model from their library (https://ollama.com/library).  llama3.1 seems to work well.
6. Choose a model when you can see a selection from the selection menu.
7. Click on Update Ollama Settings.
8. Download a whisper model by clicking Download Base English Model button.  You should see a download progress bar.  Once completed, it will download the whisper transcription model locally.  It downloads from https://huggingface.co/ggerganov/whisper.cpp/tree/main
9. Choose the transcription method.  We prefer OpenAI Whisper base.en.
10. Select an audio input device from the available options.  We prefer BlackHole 2ch after installing BlackHole and mac Loopback and mixing audio from the browser+microphone into the BlackHole 2ch.
11. Select a prompt template based on your use case.
12. Click on the Home tab
13. Start speaking or playing audio from a browser, and the application will transcribe your audio in real-time.
14. The LLM will process the transcription and provide responses.
15. Use the various buttons to control the application, including pausing/resuming transcription, clearing data, and copying responses.
16. Check the Log tab if you there is something wrong and need to debug

## Key Components

- `gui_textual.py`: Main application GUI and control logic
- `ollama_api.py`: Integration with Ollama for LLM processing
- `transcribe_audio_whisper.py`: Whisper-based audio transcription
- `transcribe_audio.py`: Vosk-based audio transcription
- `transcribe_audio_google_cloud.py`: Google Cloud Speech-to-Text integration

## Customization

- Modify prompts in `ollama_api.py` to tailor the LLM's responses to your needs
- Adjust audio processing parameters in `transcribe_audio.py` for optimal performance
- Customize the GUI layout and styling in `gui_textual.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use the files in this project except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Acknowledgements

- Vosk for offline speech recognition
- Google Cloud Speech-to-Text for online transcription
- Ollama for local LLM integration
- Textual for the TUI framework
- [pywhispercpp](https://github.com/abdeladim-s/pywhispercpp) for python bindings to whisper.cpp
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) for c++ port of OpenAI's Whisper model
- OpenAI for their Whisper model
- Meta's Llama 3.1 8B robust LLM model 
---
