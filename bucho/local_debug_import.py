import sys
import os

print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"sys.path: {sys.path}")

try:
    import ollama_api
    print(f"Successfully imported ollama_api from {ollama_api.__file__}")
except ImportError as e:
    print(f"Error importing ollama_api: {e}")

try:
    from . import gui_textual
    print(f"Successfully imported gui_textual from {gui_textual.__file__}")
except ImportError as e:
    print(f"Error importing gui_textual: {e}")

print("Import debugging complete")
