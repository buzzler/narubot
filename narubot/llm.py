import os

os.environ['OLLAMA_HOST'] = 'localhost:11434'

import ollama

from .config import Config
from .utility import available_functions
from .utility import get_tools


class LLM:
    def __init__(self, config: Config):
        self.config = config
        self.tools = get_tools()
        self.provider = (self.config.llm_provider or "ollama").lower()
        self.openai_client = None

        if self.provider == "openai":
            self._init_openai_client()

        if len(self.config.llm_system_prompt) > 0:
            self.messages = [{"role": "system", "content": self.config.llm_system_prompt}]
            self.max_messages = 33
            self.pop_at = 1
        else:
            self.messages = []
            self.max_messages = 32
            self.pop_at = 0

    def _init_openai_client(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI provider is selected, but the 'openai' package is not installed. "
                "Install it with: pip install openai"
            ) from exc

        api_key = self.config.llm_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI provider requires an API key. Set config.llm_api_key or OPENAI_API_KEY."
            )

        client_kwargs = {"api_key": api_key}
        if self.config.llm_base_url:
            client_kwargs["base_url"] = self.config.llm_base_url

        self.openai_client = OpenAI(**client_kwargs)

    def _chat_with_ollama(self) -> dict:
        return ollama.chat(
            model=self.config.llm_model,
            messages=self.messages,
            stream=False,
            tools=self.tools,
            options={"num_batch": 1},
        )

    def _chat_with_openai(self) -> tuple[dict, list]:
        response = self.openai_client.chat.completions.create(
            model=self.config.llm_model,
            messages=self.messages,
            tools=self.tools,
        )

        choice = response.choices[0].message
        message = {
            "role": "assistant",
            "content": choice.content or "",
        }

        if choice.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in choice.tool_calls
            ]

        return message, choice.tool_calls or []

    def _parse_tool_arguments(self, arguments):
        if isinstance(arguments, dict):
            return arguments

        if isinstance(arguments, str):
            try:
                import json

                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}

        return {}

    def chat(self, text: str) -> str:
        user_message = {"role": "user", "content": text}
        self.messages.append(user_message)

        if self.provider == "openai":
            msg, tool_calls = self._chat_with_openai()
        else:
            response = self._chat_with_ollama()
            msg = response["message"]
            tool_calls = msg.get("tool_calls") or []

        self.messages.append(msg)

        if tool_calls:
            for tool in tool_calls:
                if self.provider == "openai":
                    function_name = tool.function.name
                    arguments = self._parse_tool_arguments(tool.function.arguments)
                    tool_call_id = tool.id
                else:
                    function_name = tool["function"]["name"]
                    arguments = tool["function"]["arguments"]
                    tool_call_id = None

                if function_to_call := available_functions.get(function_name):
                    output = function_to_call(**arguments)
                    if self.provider == "openai":
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": str(output),
                            }
                        )
                    else:
                        self.messages.append(
                            {
                                "role": "tool",
                                "content": str(output),
                                "name": function_name,
                            }
                        )

            if self.provider == "openai":
                final_msg, _ = self._chat_with_openai()
            else:
                final_response = ollama.chat(
                    model=self.config.llm_model,
                    messages=self.messages,
                    stream=False,
                    options={"num_batch": 1},
                )
                final_msg = final_response["message"]

            msg = final_msg
            self.messages.append(msg)

        if len(self.messages) > self.max_messages:
            self.messages.pop(self.pop_at)
            self.messages.pop(self.pop_at)

        return msg.get("content", "")
