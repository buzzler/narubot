import gc
import importlib.util
import queue
import threading
import time
import wave
from .config import Config


class STT:
    def __init__(self, config: Config):
        if importlib.util.find_spec("pyaudio") is None:
            raise ModuleNotFoundError(
                "PyAudio is required for STT microphone input. "
                "Install OS audio libs then `pip install pyaudio` and retry."
            )

        import pyaudio
        import numpy as np
        from faster_whisper import WhisperModel

        if self._needs_audio_format(config):
            config.audio_format = pyaudio.paInt16

        self.config = config
        self.np = np
        self.whisper = self._create_whisper_model(WhisperModel)
        self.pyaudio = pyaudio.PyAudio()
        self.microphone = self.pyaudio.open(
            format=self.config.audio_format,
            channels=self.config.audio_channels,
            rate=self.config.audio_sample_rate,
            input=True,
            frames_per_buffer=self.config.audio_chunk,
        )

        self.audio_frames = []
        self.silent_chunks = 0
        self.speaking = False
        self.activated = False
        self.llm = None
        self.tts = None

        self.running = False
        self.utterance_queue = queue.Queue(maxsize=self.config.stt_queue_max_size)
        self.worker_thread = None
        self.tts_playing = threading.Event()
        self.capture_paused_for_tts = False

    def _create_whisper_model(self, whisper_model_cls):
        try:
            return whisper_model_cls(
                self.config.stt_model,
                device=self.config.device,
                compute_type=self.config.stt_compute_type,
                cpu_threads=8,
                num_workers=1,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            if not str(self.config.device).startswith("cuda") or not self._is_cuda_runtime_error(exc):
                raise

            print(
                "STT CUDA initialization failed. Falling back to CPU. "
                f"Reason: {exc}"
            )
            self.config.device = "cpu"
            if self.config.stt_compute_type in {"float16", "int8_float16"}:
                self.config.stt_compute_type = "int8"
            return whisper_model_cls(
                self.config.stt_model,
                device=self.config.device,
                compute_type=self.config.stt_compute_type,
                cpu_threads=8,
                num_workers=1,
            )

    @staticmethod
    def _is_cuda_runtime_error(exc: Exception) -> bool:
        message = str(exc).lower()
        cuda_error_markers = (
            "cublas",
            "cuda",
            "cudnn",
            ".dll",
            "cannot be loaded",
            "failed to load",
            "runtimeerror: library",
        )
        return any(marker in message for marker in cuda_error_markers)

    @staticmethod
    def _needs_audio_format(config: Config) -> bool:
        return config.audio_format is None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _is_silent(self, data: bytes) -> bool:
        return self.np.abs(self.np.frombuffer(data, dtype=self.np.int16)).mean() < self.config.silence_threshold

    @staticmethod
    def _normalize_command_text(text: str) -> str:
        return "".join(ch for ch in text.casefold() if ch.isalnum())

    def _is_start_of_speech(self, text: str) -> tuple[bool, str]:
        normalized_text = self._normalize_command_text(text)
        for command in self.config.stt_start_commands:
            if text.startswith(command):
                return True, text.removeprefix(command).strip()
            normalized_command = self._normalize_command_text(command)
            if normalized_command and normalized_text.startswith(normalized_command):
                return True, text

        for command in self.config.stt_magic_commands:
            if text.find(command) != -1:
                return True, text
            normalized_command = self._normalize_command_text(command)
            if normalized_command and normalized_command in normalized_text:
                return True, text
        return False, ""

    def _is_end_of_speech(self, text: str) -> bool:
        normalized_text = self._normalize_command_text(text)
        for command in self.config.stt_stop_commands:
            if text.startswith(command):
                return True
            normalized_command = self._normalize_command_text(command)
            if normalized_command and normalized_text.startswith(normalized_command):
                return True
        return False

    def _play_effect(self, effect_path: str):
        if not self.config.tts_enabled:
            return

        should_pause_stt = self.config.stt_pause_during_tts
        if should_pause_stt:
            self.tts_playing.set()

        try:
            with wave.open(effect_path, "rb") as wav:
                stream = self.pyaudio.open(
                    format=self.pyaudio.get_format_from_width(wav.getsampwidth()),
                    channels=wav.getnchannels(),
                    rate=wav.getframerate(),
                    output=True,
                )
                data = wav.readframes(self.config.audio_chunk)
                while data:
                    stream.write(data)
                    data = wav.readframes(self.config.audio_chunk)
                stream.stop_stream()
                stream.close()
        finally:
            if should_pause_stt:
                self.tts_playing.clear()

    def _activate(self):
        from .llm import LLM
        from .tts import TTS

        self.activated = True
        self._print_listening_status()
        self.llm = LLM(self.config)
        self.tts = TTS(self.config) if self.config.tts_enabled else None
        self._play_effect(r"asset/start.wav")

    def _deactivate(self):
        self._play_effect(r"asset/end.wav")
        self.activated = False
        self._print_listening_status()
        del self.llm
        if self.tts is not None:
            del self.tts
        self.llm = None
        self.tts = None
        gc.collect()

    def _frames_to_audio_array(self, frames: list[bytes]):
        if not frames:
            return None

        pcm = self.np.frombuffer(b"".join(frames), dtype=self.np.int16)
        if pcm.size == 0:
            return None

        return pcm.astype(self.np.float32) / 32768.0

    def _speech_to_text_from_frames(self, frames: list[bytes]) -> str:
        audio = self._frames_to_audio_array(frames)
        if audio is None:
            return ""

        transcribe_kwargs = {
            "language": self.config.stt_language,
            "beam_size": self.config.stt_beam_size,
            "vad_filter": self.config.stt_vad_filter,
        }
        if self.config.stt_vad_filter:
            transcribe_kwargs["vad_parameters"] = {
                "min_silence_duration_ms": self.config.stt_vad_min_silence_ms,
            }

        segments, _ = self.whisper.transcribe(audio, **transcribe_kwargs)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _clear_capture_buffers(self) -> None:
        self.audio_frames = []
        self.silent_chunks = 0
        self.speaking = False

    def _print_listening_status(self) -> None:
        if self.activated:
            print("Listening (in conversation)...")
        else:
            print("Listening (standby)...")

    def _flush_stream(self) -> None:
        self._clear_capture_buffers()
        self._print_listening_status()

    def _enqueue_utterance(self, frames: list[bytes]) -> None:
        if not frames:
            return

        try:
            self.utterance_queue.put_nowait(frames)
        except queue.Full:
            print("STT queue is full. Dropping oldest utterance.")
            try:
                self.utterance_queue.get_nowait()
            except queue.Empty:
                pass
            self.utterance_queue.put_nowait(frames)

    def _build_llm_batch_text(self, first_text: str) -> str:
        if not self.config.llm_batch_enabled:
            return first_text

        batch = [first_text]
        deadline = time.monotonic() + max(0.0, float(self.config.llm_batch_window_sec))
        max_items = max(1, int(self.config.llm_batch_max_items))

        while len(batch) < max_items and time.monotonic() < deadline:
            timeout = max(0.0, deadline - time.monotonic())
            if timeout == 0.0:
                break

            try:
                frames = self.utterance_queue.get(timeout=timeout)
            except queue.Empty:
                break

            next_text = self._speech_to_text_from_frames(frames)
            if next_text:
                print("User (queued):", next_text)
                batch.append(next_text)

        if len(batch) == 1:
            return batch[0]

        return "\n".join(batch)

    def _wait_for_user_silence(self) -> None:
        if (not self.config.tts_enabled) or (not self.config.tts_wait_for_user_silence):
            return

        deadline = time.monotonic() + max(0.0, float(self.config.tts_wait_timeout_sec))
        while self.running and self.speaking and time.monotonic() < deadline:
            time.sleep(0.01)

    def _speak_text(self, text: str) -> None:
        if (not self.config.tts_enabled) or (not text):
            return

        self._wait_for_user_silence()

        should_pause_stt = self.config.stt_pause_during_tts
        if should_pause_stt:
            self.tts_playing.set()

        try:
            self.tts.text_to_speech(text)
        finally:
            if should_pause_stt:
                self.tts_playing.clear()

    def _process_utterance(self, frames: list[bytes]) -> None:
        text = self._speech_to_text_from_frames(frames)
        if not text:
            return

        if not self.activated:
            is_start, start_text = self._is_start_of_speech(text)
            if not is_start:
                print("Not activated:", text)
                return

            self._activate()
            self._speak_text("Starting conversation.")
            text = start_text if start_text else ""

        if text and self._is_end_of_speech(text):
            self._deactivate()
            return

        if not self.activated or not text:
            return

        print("User:", text)
        batched_text = self._build_llm_batch_text(text)
        self._play_effect(r"asset/process.wav")
        response = self.llm.chat(batched_text).strip()
        print("Assistant:", response)
        self._speak_text(response)

    def _worker_loop(self) -> None:
        while self.running:
            try:
                frames = self.utterance_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._process_utterance(frames)
            except Exception as exc:
                print(f"Worker error: {exc}")

    def process_audio_loop(self) -> None:
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, name="stt-worker", daemon=True)
        self.worker_thread.start()
        self._flush_stream()

        while self.running:
            data = self.microphone.read(self.config.audio_chunk, exception_on_overflow=False)

            if self.tts_playing.is_set():
                if not self.capture_paused_for_tts:
                    self._clear_capture_buffers()
                    self.capture_paused_for_tts = True
                continue

            if self.capture_paused_for_tts:
                self._clear_capture_buffers()
                self.capture_paused_for_tts = False

            if self._is_silent(data):
                self.silent_chunks += 1
                silence_frames = self.config.silence_limit * self.config.audio_sample_rate / self.config.audio_chunk
                if self.speaking and self.silent_chunks > silence_frames:
                    self.speaking = False
                    utterance = self.audio_frames
                    self._flush_stream()
                    self._enqueue_utterance(utterance)
            else:
                self.silent_chunks = 0
                if not self.speaking:
                    self.speaking = True
                self.audio_frames.append(data)

    def close(self):
        print("Closing...")
        self.running = False

        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None

        if self.microphone is not None:
            self.microphone.stop_stream()
            self.microphone.close()
            self.microphone = None

        if self.pyaudio is not None:
            self.pyaudio.terminate()
            self.pyaudio = None
