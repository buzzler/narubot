import os
os.environ['OLLAMA_HOST'] = 'localhost:11434'
import ollama
from .config import Config
from .utility import get_tools
from .utility import available_functions

class LLM:
    def __init__(self, config: Config):
        self.config = config
        self.tools = get_tools()
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
        response : ollama.ChatResponse = ollama.chat(
            model=self.config.llm_model,
            messages=self.messages,
            stream=False,
            tools=self.tools,
            options={"num_batch": 1}
        )
        msg = response['message']
        self.messages.append(msg)
        tool_calls = response['message'].get('tool_calls')
        if tool_calls is not None and len(tool_calls) > 0:
            for tool in tool_calls:
                if function_to_call := available_functions.get(tool['function']['name']):
                    output = function_to_call(**tool['function']['arguments'])
                    self.messages.append({'role': 'tool', 'content': str(output), 'name': tool['function']['name']})
            final_response : ollama.ChatResponse = ollama.chat(
                model=self.config.llm_model, 
                messages=self.messages, 
                stream=False,
                options={"num_batch": 1})
            msg = final_response['message']
            self.messages.append(msg)

        if len(self.messages) > self.max_messages:
            self.messages.pop(self.pop_at)
            self.messages.pop(self.pop_at)
        return msg['content']