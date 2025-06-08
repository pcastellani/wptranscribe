# wptranscribe

`wptranscribe` is a Python command-line script for audio transcription using OpenAI's Whisper model.
Whisper + Python -> transcribe

## Features

* **Whisper Model Integration:** Interfaces with Hugging Face's Whisper implementation.
* **Low VRAM Mode:** Optionally offloads model weights using `accelerate` for systems with limited GPU memory.
* **Audio Chunking:** Splits long audio into manageable, overlapping segments.
* **Repetition Removal:** Removes redundant text from overlapping segments using fuzzy matching.
* **Custom Models:** Supports loading custom-trained Whisper model weights.
* **CUDA/CPU Support:** Can use either GPU (CUDA) if available or CPU.
* **Resume Transcription:** Allows resuming transcription from a specified audio chunk.
* **Command Line Feedback** Provides real-time progress updates during transcription.

## Getting Started

### Prerequisites

* All Python dependencies listed in `requirements_cpu.txt` or:
* For CUDA support (optional) use the reference `requirements.txt` adjusting the CUDA (cu### in the URL) version as needed.
* * The cudnn line is an example in case a specific version of it is needed.
* `ffmpeg` (external, for audio format conversion if your file is not 16KHz mono .wav)

### Usage

```bash
python transcribe.py --audio-path <path_to_audio_file> --language <language_code> [options]
