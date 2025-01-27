from melo.api import TTS as Melo
from .config import Config

class TTS:
    def __init__(self, config : Config):
        self.config = config
        self.model = Melo(language=self.config.tts_language, device=self.config.device)
        self.speaker_id = self.model.hps.data.spk2id[self.config.tts_language]

    def text_to_file(self, text: str, file_path: str) -> None:
        self.model.tts_to_file(text, self.speaker_id, 
            speed=self.config.tts_speed, 
            quiet=True, 
            output_path=file_path)

    def text_to_speech(self, text: str) -> None:
        self.model.tts_to_file(text, self.speaker_id, 
            speed=self.config.tts_speed, 
            quiet=True, 
            play_audio=True)
