import pyaudio

class Config:
    def __init__(self, 
            device = "cpu", 
            stt_model = "large-v2", 
            stt_compute_type = "float32", 
            stt_language = "ko",
            stt_start_commands = ["시작"],
            stt_magic_commands = [],
            stt_stop_commands = ["종료"],
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
        self.stt_model = stt_model  # The model size to use for transcription.
        self.stt_compute_type = stt_compute_type  # The compute type to use for the model.
        self.stt_language = stt_language # The language to use for transcription.
        self.stt_start_commands = stt_start_commands  # Commands to start the transcription process.
        self.stt_magic_commands = stt_magic_commands  # Magic commands to trigger specific actions during transcription.
        self.stt_stop_commands = stt_stop_commands  # Commands to stop the transcription process.
        self.llm_model = llm_model  # The Ollama model to use for chatbot interactions.
        self.llm_system_prompt = llm_system_prompt  # System prompt for Ollama.
        self.wav_file = wav_file  # File to save audio data as WAV.
        self.audio_chunk = audio_chunk  # Number of frames per buffer for PyAudio.
        self.audio_format = audio_format  # Audio format for recording.
        self.audio_channels = audio_channels  # Number of channels for recording.
        self.audio_sample_rate = audio_sample_rate # Sampling rate for recording.
        self.silence_threshold = silence_threshold  # Threshold to detect silence in audio data.
        self.silence_limit = silence_limit  # Maximum number of silent chunks before considering speech as ended.
        self.tts_speed = tts_speed  # Speed for text-to-speech conversion.
        self.tts_language = tts_language  # Language for text-to-speech conversion.
