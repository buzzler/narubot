from __future__ import annotations

import importlib.util


def _resolve_default_audio_format() -> int | None:
    """Resolve PyAudio sample format lazily so Config can be imported without PyAudio."""
    if importlib.util.find_spec("pyaudio") is None:
        return None

    import pyaudio

    return pyaudio.paInt16


def _resolve_device(device: str) -> str:
    """Select a runtime-safe inference device.

    Falls back to CPU when CUDA is requested but unavailable in the current
    Python environment.
    """
    if device != "cuda":
        return device

    if importlib.util.find_spec("torch") is None:
        return "cpu"

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Config:
    def __init__(self,
            device = "cpu",
            stt_model = "large-v3",
            stt_compute_type = "int8_float16",
            stt_language = "ko",
            stt_beam_size = 5,
            stt_vad_filter = True,
            stt_vad_min_silence_ms = 300,
            stt_start_commands = ["\uc2dc\uc791"],
            stt_magic_commands = [],
            stt_stop_commands = ["\uc885\ub8cc"],
            llm_provider = "ollama",
            llm_model = "mistral:7b",
            llm_system_prompt = "",
            llm_api_key = None,
            llm_base_url = None,
            llm_batch_enabled = True,
            llm_batch_window_sec = 0.35,
            llm_batch_max_items = 3,
            wav_file = "conversation.wav",
            audio_chunk = 1024,
            audio_format = None,
            audio_channels = 1,
            audio_sample_rate = 16000,
            silence_threshold = 500,
            silence_limit = 1,
            stt_queue_max_size = 32,
            stt_pause_during_tts = True,
            tts_wait_for_user_silence = True,
            tts_wait_timeout_sec = 2.0,
            tts_speed = 1.3,
            tts_language = "KR"):
        self.device = _resolve_device(device)  # The device to use for inference.
        self.stt_model = stt_model  # The model size to use for transcription.
        self.stt_compute_type = stt_compute_type  # The compute type to use for the model.
        self.stt_language = stt_language  # The language to use for transcription.
        self.stt_beam_size = stt_beam_size  # Beam width for decoding quality/speed tradeoff.
        self.stt_vad_filter = stt_vad_filter  # Whether to apply model-side VAD filtering.
        self.stt_vad_min_silence_ms = stt_vad_min_silence_ms  # Minimum silence for VAD split.
        self.stt_start_commands = stt_start_commands  # Commands to start the transcription process.
        self.stt_magic_commands = stt_magic_commands  # Magic commands to trigger specific actions during transcription.
        self.stt_stop_commands = stt_stop_commands  # Commands to stop the transcription process.
        self.llm_provider = llm_provider  # LLM provider to use ("ollama" or "openai").
        self.llm_model = llm_model  # Model name used by the selected LLM provider.
        self.llm_system_prompt = llm_system_prompt  # System prompt for LLM.
        self.llm_api_key = llm_api_key  # API key for cloud providers such as OpenAI.
        self.llm_base_url = llm_base_url  # Optional custom base URL (e.g. OpenAI-compatible endpoint).
        self.llm_batch_enabled = llm_batch_enabled  # Whether to combine queued utterances into one LLM turn.
        self.llm_batch_window_sec = llm_batch_window_sec  # Time window to collect extra queued utterances.
        self.llm_batch_max_items = llm_batch_max_items  # Maximum utterances to merge in a single LLM turn.
        self.wav_file = wav_file  # File to save audio data as WAV.
        self.audio_chunk = audio_chunk  # Number of frames per buffer for PyAudio.
        self.audio_format = audio_format if audio_format is not None else _resolve_default_audio_format()  # Audio format for recording.
        self.audio_channels = audio_channels  # Number of channels for recording.
        self.audio_sample_rate = audio_sample_rate  # Sampling rate for recording.
        self.silence_threshold = silence_threshold  # Threshold to detect silence in audio data.
        self.silence_limit = silence_limit  # Maximum number of silent chunks before considering speech as ended.
        self.stt_queue_max_size = stt_queue_max_size  # Max buffered utterances while worker is busy.
        self.stt_pause_during_tts = stt_pause_during_tts  # Pause STT capture while speaker output is playing.
        self.tts_wait_for_user_silence = tts_wait_for_user_silence  # Wait until user speech ends before TTS output.
        self.tts_wait_timeout_sec = tts_wait_timeout_sec  # Max wait time before speaking anyway.
        self.tts_speed = tts_speed  # Speed for text-to-speech conversion.
        self.tts_language = tts_language  # Language for text-to-speech conversion.
