import os
os.environ['OLLAMA_HOST'] = 'localhost:11434'
import ollama
from .config import Config

class LLM:
    def __init__(self, config: Config):
        self.config = config
        if len(self.config.llm_system_prompt) > 0:
            self.messages = [{"role": "system", "content": self.config.llm_system_prompt}]
            self.max_messages = 33
            self.pop_at = 1
        else:
            self.messages = []
            self.max_messages = 32
            self.pop_at = 0

    def chat(self, text:str) -> str:
        msg = {"role": "user", "content": text}
        self.messages.append(msg)
        response = ollama.chat(
            model=self.config.llm_model,
            messages=self.messages,
            stream=False,
        )
        msg = response['message']
        self.messages.append(msg)
        if len(self.messages) > self.max_messages:
            self.messages.pop(self.pop_at)
            self.messages.pop(self.pop_at)
        return msg['content']