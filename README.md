## NaruBot

NaruBot is a real-time conversational AI system that integrates Speech-to-Text (STT), Text-to-Speech (TTS), and Large Language Models (LLMs) to enable natural, low-latency voice interaction.

The project focuses on designing an event-driven pipeline that handles audio input, language understanding, and speech synthesis in real time. While the current implementation runs fully in software, the system is intentionally designed with constrained, hardware-like interfaces in mind.

The long-term goal is to extend Narubot to a custom hardware agent cartridge inspired by retro console environments, exploring how modern AI systems can operate under limited I/O, timing, and interaction constraints.

## Runtime notes (Python 3.12 + CUDA)

- Python `3.12` is supported.
- For CUDA acceleration with the latest NVIDIA drivers/toolkit, install PyTorch from the official CUDA wheel index instead of relying on a fixed `+cuXXX` pin in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Example: install latest PyTorch CUDA wheels (adjust CUDA index if needed)
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchaudio torchvision
```

- `PyAudio` also requires system audio libraries. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev
pip install pyaudio
```
