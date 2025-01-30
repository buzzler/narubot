import gc
import pyaudio
import numpy as np
import wave
from faster_whisper import WhisperModel
from .config import Config
from .tts import TTS
from .llm import LLM

class STT:
    def __init__(self, config: Config):
        self.config = config
        self.whisper = None
        self.pyaudio = pyaudio.PyAudio()
        self.microphone = self.pyaudio.open(
            format=self.config.audio_format,
            channels=self.config.audio_channels,
            rate=self.config.audio_sample_rate,
            input=True,
            frames_per_buffer=self.config.audio_chunk)
        self.audio_frames = []
        self.silent_chunks = 0
        self.speaking = False
        self.activated = False
        self.llm = None
        self.tts = None
        self._init_whisper()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _init_whisper(self):
        if self.whisper is not None:
            del self.whisper
            self.whisper = None
        if self.activated:
            self.whisper = WhisperModel(self.config.stt_model, device=self.config.device, compute_type=self.config.stt_compute_type)
        else:
            self.whisper = WhisperModel(self.config.stt_standby_model, device=self.config.device, compute_type=self.config.stt_compute_type)

    def _is_silent(self, data: bytes) -> bool:
        return np.abs(np.frombuffer(data, dtype=np.int16)).mean() < self.config.silence_threshold
    
    def _is_start_of_speech(self, text: str) -> tuple[bool, str]:
        for command in self.config.stt_start_commands:
            if text.startswith(command):
                return True, text.removeprefix(command).strip()
        for command in self.config.stt_magic_commands:
            if text.find(command) != -1:
                return True, text
        return False, ""

    def _is_end_of_speech(self, text: str) -> bool:
        for command in self.config.stt_stop_commands:
            if text.startswith(command):
                return True
        return False
    
    def _play_effect(self, effect_path: str):
        with wave.open(effect_path,"rb") as wav:
            stream = self.pyaudio.open(format = self.pyaudio.get_format_from_width(wav.getsampwidth()), 
                                   channels = wav.getnchannels(),
                                   rate = wav.getframerate(),
                                   output = True)
            data = wav.readframes(self.config.audio_chunk)
            while data:
                stream.write(data)
                data = wav.readframes(self.config.audio_chunk)
            stream.stop_stream()
            stream.close()

    def _activate(self):
        self.activated = True
        self.llm = LLM(self.config)
        self.tts = TTS(self.config)
        self._play_effect(r'asset/start.wav')
    
    def _deactivate(self):
        self._play_effect(r'asset/end.wav')
        self.activated = False
        del self.llm
        del self.tts
        self.llm = None
        self.tts = None
        gc.collect()

    def _save_wav(self) -> None:
        with wave.open(self.config.wav_file, 'wb') as wf:
            wf.setnchannels(self.config.audio_channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.config.audio_format))
            wf.setframerate(self.config.audio_sample_rate)
            wf.writeframes(b''.join(self.audio_frames))
    
    def _speech_to_text(self) -> str:
        if self.activated:
            lang = self.config.stt_language
        else:
            lang = self.config.stt_standby_language
        with open(self.config.wav_file, "rb") as audio_file:
            segments, _ = self.whisper.transcribe(audio_file, language=lang)
            segments = list(segments)
            return str.join(" ", [segment.text.strip() for segment in segments])
    
    def _flush_stream(self) -> None:
        self.audio_frames = []
        self.silent_chunks = 0
        if self.activated:
            self._play_effect(r'asset/turn.wav')
        print("Listening...")

    def process_audio_loop(self) -> None:
        self._flush_stream()
        while True:
            data = self.microphone.read(self.config.audio_chunk)
            if self._is_silent(data):
                self.silent_chunks += 1
                if self.speaking and self.silent_chunks > self.config.silence_limit * self.config.audio_sample_rate / self.config.audio_chunk:
                    self.speaking = False
                    self._save_wav()
                    text = self._speech_to_text()
                    print("User:", text)

                    if not self.activated:
                        is_start, text = self._is_start_of_speech(text)
                        if is_start:
                            self._activate()
                            self._init_whisper()

                    if self.activated:
                        if self._is_end_of_speech(text):
                            self._deactivate()
                            self._init_whisper()

                    if self.activated:
                        self._play_effect(r'asset/process.wav')
                        response = self.llm.chat(text)
                        print("Assistant:", response)
                        self.tts.text_to_speech(response)

                    self._flush_stream()
            else:
                self.silent_chunks = 0
                if not self.speaking:
                    self.speaking = True
                self.audio_frames.append(data)

    def close(self):
        print("Closing...")
        self.pyaudio.close(self.microphone)
