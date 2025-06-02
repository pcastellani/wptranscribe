# wptranscribe

`wptranscribe` is a Python command-line script for audio transcription using OpenAI's Whisper model.

## Features

* **Whisper Model:** Leverages Hugging Face Transformers' Whisper implementation.
* **Low VRAM Mode:** Optionally offloads model weights using `accelerate` for limited GPU memory systems.
* **Audio Chunking:** Splits long audio into manageable, overlapping segments.
* **Repetition Removal:** Detects and removes redundant text from overlapping chunks.
* **Custom Models:** Supports loading custom-trained Whisper model weights.
* **CUDA/CPU Support:** Can use either GPU (CUDA) if available or CPU.
* **Resume Transcription:** Start transcription from a specified chunk.

## Getting Started

### Prerequisites

* All Python dependencies listed in `requirements.txt` (ajust the CUDA version if needed)
* `ffmpeg` (for audio format conversion if your file is not 16KHz mono .wav)

### Usage

```bash
python transcribe.py --audio-path <path_to_audio_file> --language <language_code> [options]
