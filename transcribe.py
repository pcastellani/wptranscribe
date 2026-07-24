import torch
import os
import sys
import math
import argparse
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import difflib
import importlib

# Enable multi-threading for PyTorch
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())

# Enable synchronous debug mode to ensure CUDA OOM errors are raised in the main thread (only if CUDA is available)
if torch.cuda.is_available() and hasattr(torch.cuda, "set_sync_debug_mode"):
	torch.cuda.set_sync_debug_mode('default')

def get_unique_output_path(path):
	base, ext = os.path.splitext(path)
	counter = 1
	while os.path.exists(path):
		path = f"{base}_{counter}{ext}"
		counter += 1
	return path

def get_model_name(size: str) -> str:
	return f"openai/whisper-{size}"

def load_custom_whisper_model(model_path, device, model_size, lowvram=False):
	try:
		model_name = get_model_name(model_size)
		print(f"Initializing Whisper model: {model_name}...")

		if lowvram:
			try:
				importlib.import_module("accelerate")
				print("Using Accelerate to offload weights to CPU.")

				model = WhisperForConditionalGeneration.from_pretrained(
					model_name,
					device_map="auto",
					dtype=torch.float16
				)
			except ImportError:
				print("⚠️ Low VRAM mode requested, but 'accelerate' is not available. Proceeding with standard model loading.")
				model = WhisperForConditionalGeneration.from_pretrained(model_name)
		else:
			model = WhisperForConditionalGeneration.from_pretrained(model_name)

		if model_path and os.path.isfile(model_path):
			print(f"Loading custom weights from: {model_path}")
			state_dict = torch.load(model_path, map_location=torch.device(device))
			model.load_state_dict(state_dict, strict=False)
			print("Custom weights loaded successfully.")

		if not lowvram:
			model = model.to(device)

		print(f"Loaded {model_name} model successfully on {device}!")
		return model

	except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
		if device == "cuda":
			print("\n\n🚨 CUDA Out of Memory Error! Clearing GPU memory and exiting... 🚨\n", flush=True)
			torch.cuda.empty_cache()
		os._exit(1)

def split_audio_into_chunks(audio, sample_rate, chunk_duration=30, overlap=2):
	chunk_size = chunk_duration * sample_rate
	overlap_size = overlap * sample_rate

	chunks = []
	for start in range(0, len(audio), chunk_size - overlap_size):
		end = start + chunk_size
		chunk = audio[start:end]
		chunks.append(chunk)
		if end >= len(audio):
			break
	return chunks

def transcribe_chunk(model, chunk, processor, device, language):
	try:
		inputs = processor(chunk, return_tensors="pt", sampling_rate=16000, return_attention_mask=True)
		input_features = inputs.input_features.to(device=device)
		attention_mask = inputs.attention_mask.to(device=device)

		with torch.autocast(device_type=device):  # Mixed Precision enabled for both CUDA and CPU
			predicted_ids = model.generate(
				input_features,
				attention_mask=attention_mask,
				num_beams=5,
				no_repeat_ngram_size=2,
				language=language,
				task="transcribe"
			)

		transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		return transcription

	except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
		if device == "cuda":
			print("\n\n🚨 CUDA Out of Memory Error! Clearing GPU memory and exiting... 🚨\n", flush=True)
			torch.cuda.empty_cache()
		os._exit(1)  # Using os._exit to ensure an immediate exit instead
	except Exception as e:
		print(f"Error during transcription: {e}")
		sys.exit(1)

def transcribe_audio(model, processor, audio_path, output_path, device, language, chunk_duration, overlap, dedup_window, dedup_threshold, initial_chunk):
	try:
		if not os.path.isfile(audio_path):
			raise FileNotFoundError(f"Audio file not found: {audio_path}")

		ext = os.path.splitext(audio_path)[1].lower()
		if ext != '.wav':
			print("Invalid file format")
			print("Convert other formats to 16KHz mono .wav")
			print("Example: ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav")
			sys.exit(1)

		print(f"Loading audio file using soundfile: {audio_path}")
		audio, sr = sf.read(audio_path)

		if audio.ndim > 1:
			channels = audio.shape[1]
			print(f"⚠️ Input audio has {channels} channels. Converting to mono by averaging channels.")
			audio = audio.mean(axis=1)

		if sr != 16000:
			print(f"Resampling audio from {sr} Hz to 16000 Hz")
			gcd = math.gcd(sr, 16000)
			up = 16000 // gcd
			down = sr // gcd
			scipy_signal = importlib.import_module("scipy.signal")
			audio = scipy_signal.resample_poly(audio, up, down).astype("float32")

		print("Audio loaded successfully!")

		print("Splitting audio into chunks...")
		chunks = split_audio_into_chunks(audio, 16000, chunk_duration=chunk_duration, overlap=overlap)
		total_chunks = len(chunks)
		if initial_chunk > total_chunks:
			print(f"Initial chunk {initial_chunk} exceeds total chunks ({total_chunks}). Nothing to do.")
			sys.exit(1)

		print(f"Audio split into {total_chunks} chunks of {chunk_duration}s each with {overlap}s overlap")

		if output_path is None:
			base_filename = os.path.splitext(os.path.basename(audio_path))[0]
			output_path = f"{base_filename}_transcription.txt"
			print(f"No output path provided. Defaulting to: {output_path}")

		output_path = get_unique_output_path(output_path)
		print(f"Writing transcription to: {output_path}")

		with open(output_path, 'w', encoding='utf-8') as file:
			previous_transcription = ""
			for offset, chunk in enumerate(chunks[initial_chunk - 1 :], start=initial_chunk):
				print(f"Transcribing chunk {offset} of {total_chunks}...")
				try:
					transcription = transcribe_chunk(model, chunk, processor, device, language=language)
					transcription = remove_repetitions(previous_transcription, transcription, dedup_window, dedup_threshold)
					print(f"Chunk {offset} transcription: {transcription}")
					file.write(transcription + ' ')
					file.flush()
					previous_transcription += transcription

				except Exception as e:
					print(f"Error transcribing chunk {offset}: {e}")
		
		print(f"Transcription saved to: {output_path}")

	except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
		if device == "cuda":
			print("\n\n🚨 CUDA Out of Memory Error! Clearing GPU memory and exiting... 🚨\n", flush=True)
			torch.cuda.empty_cache()
		os._exit(1)
	except Exception as e:
		print(f"Error during transcription: {e}")
		sys.exit(1)

def remove_repetitions(previous_text, current_text, window, min_ratio):
	if not previous_text:
		return current_text

	previous_tail = previous_text[-window:]

	matcher = difflib.SequenceMatcher(None, previous_tail, current_text)
	match = matcher.find_longest_match(0, len(previous_tail), 0, len(current_text))

	# If the match starts at the end of previous_text and at the beginning of current_text
	if match.b == 0:
		ratio = matcher.ratio()
		if ratio >= min_ratio:
			return current_text[match.size:]

	return current_text

def main():
	parser = argparse.ArgumentParser(description="Transcribe an audio file using a custom Whisper model with low VRAM support.")
	parser.add_argument('-m', '--model-path', type=str, default=None, help="Path to the custom Whisper model file (.bin).")
	parser.add_argument('-a', '--audio-path', type=str, required=True, help="Path to the input audio file to transcribe.")
	parser.add_argument('-o', '--output-path', type=str, default=None, help="Path to save the transcription output.")
	parser.add_argument('-l', '--language', type=str, required=True, help="Language to use for transcription.")
	parser.add_argument('-s', '--model-size', type=str, default="small", choices=['tiny', 'small', 'medium', 'large'], help="Size of the Whisper model to use (default: 'small').")
	parser.add_argument('--dedup-window', type=int, default=200, help="Window size (in characters) for overlap detection.")
	parser.add_argument('--dedup-threshold', type=float, default=0.9, help="Similarity threshold for overlap removal.")
	parser.add_argument('--chunk-duration', type=int, default=30, help="Chunk duration (in seconds) for audio splitting.")
	parser.add_argument('--overlap', type=int, default=2, help="Overlap (in seconds) between chunks.")
	parser.add_argument('--initial_chunk', type=int, default=1, help="Chunk number (1-indexed) to start transcription from. Default is 1 (start from beginning).")
	parser.add_argument('--use-cuda', action='store_true', help="Use CUDA if available, otherwise default to CPU.")
	parser.add_argument('--lowvram', action='store_true', help="Enable low VRAM mode using Accelerate and CPU offloading.")

	args = parser.parse_args()

	device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
	if args.use_cuda and device != "cuda":
		print("⚠️ CUDA requested but not available. Defaulting to CPU.")
	
	print(f"Using device: {device}")

	if args.lowvram and device != "cuda":
		print("⚠️ Low VRAM mode requested, but CUDA is not in use. Ignoring.")
		args.lowvram = False

	model = load_custom_whisper_model(
		model_path=args.model_path,
		device=device,
		model_size=args.model_size,
		lowvram=args.lowvram
	)

	processor = WhisperProcessor.from_pretrained(get_model_name(args.model_size))

	try:
		processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
		print(f"✅ Language '{args.language}' supported by model.")
	except Exception as e:
		print(f"❌ Error verifying language '{args.language}': {e}")
		parser.error(f"Unsupported language code '{args.language}'.")

	transcribe_audio(
		model=model,
		processor=processor,
		audio_path=args.audio_path,
		output_path=args.output_path,
		device=device,
		language=args.language,
		chunk_duration=args.chunk_duration,
		overlap=args.overlap,
		dedup_window=args.dedup_window,
		dedup_threshold=args.dedup_threshold,
		initial_chunk=args.initial_chunk
	)

if __name__ == "__main__":
	main()
