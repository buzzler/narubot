import pyaudio

class Config:
    def __init__(self, 
            device = "cpu", 
            stt_model = "medium", 
            stt_compute_type = "float32", 
            stt_language = "ko",
            llm_model = "mistral:7b",
            llm_system_prompt = "",
            wav_file = "conversation.wav",
            audio_chunk = 1024,
            audio_format = pyaudio.paInt16,
            audio_channels = 1,
            audio_sample_rate = 16000,
            silence_threshold= 500,
            silence_limit = 1,
            tts_speed = 1.3,
            tts_language = "KR"):
        self.device = device # The device to use for inference.
        self.whisper_model = stt_model  # The model size to use for transcription.
        self.whisper_compute_type = stt_compute_type  # The compute type to use for the model.
        self.whisper_language = stt_language # The language to use for transcription.
        self.ollama_model = llm_model  # The Ollama model to use for chatbot interactions.
        self.ollama_system_prompt = llm_system_prompt  # System prompt for Ollama.
        self.wav_file = wav_file  # File to save audio data as WAV.
        self.audio_chunk = audio_chunk  # Number of frames per buffer for PyAudio.
        self.audio_format = audio_format  # Audio format for recording.
        self.audio_channels = audio_channels  # Number of channels for recording.
        self.audio_sample_rate = audio_sample_rate # Sampling rate for recording.
        self.silence_threshold = silence_threshold  # Threshold to detect silence in audio data.
        self.silence_limit = silence_limit  # Maximum number of silent chunks before considering speech as ended.
        self.tts_speed = tts_speed  # Speed for text-to-speech conversion.
        self.tts_language = tts_language  # Language for text-to-speech conversion.
