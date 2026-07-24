# wptranscribe

`wptranscribe` is a Python command-line script for local audio transcription using OpenAI's Whisper model.

Whisper + Python -> transcribe

## Why wptranscribe?

Unlike cloud transcription services, **the audio is processed locally**.

- Audio files are **never uploaded** to any server.
- Transcription happens entirely on the local machine.
- Generated transcripts remain local.
- Internet access is only required the first time a Whisper model is downloaded (or when downloading a different model).
- Once the model is cached, transcription works completely offline.
- No service-imposed file size limits or monthly transcription quotas.

This makes `wptranscribe` suitable for confidential recordings, internal company meetings, research interviews, personal archives, and other privacy-sensitive workloads.

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
* `ffmpeg` (external, for audio format conversion if your file is not 16KHz mono .wav)

### Usage

```bash
python transcribe.py --audio-path <path_to_audio_file> --language <language_code> [options]
```

**Warning: Extra Disk Usage in Low VRAM Mode**

Whisper models are always cached locally by Hugging Face the first time.
However, when run with `--lowvram`, `snapshot_download()` will be invoked, which stores a **full copy of the model repository** in the cache directory.
This will take additional disk space compared to normal mode.

The cache is located by default under:
- `~/.cache/huggingface/hub/`

To free this space later, you can remove each of the downloaded snapshots like the following example (will be re-download if needed):
```bash
rm -rf ~/.cache/huggingface/hub/models--openai--whisper-small
```
