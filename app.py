import narubot

config = narubot.Config(
    device = "cuda",
    stt_start_commands=["이봐", "챗봇", "어시스턴트", "Assistant", "안녕", "Hey"],
    stt_stop_commands=["종료", "끝", "그만", "멈춰", "잘가", "조용", "시끄러"],
    stt_magic_commands=["시리"],
    llm_provider = "ollama",  # "ollama" or "openai"
    llm_model = "qwen2.5:7b",
    llm_api_key = None,  # Set for OpenAI (or use OPENAI_API_KEY env var)
    llm_base_url = "https://api.openai.com/v1",  # Optional OpenAI-compatible base URL
    llm_system_prompt = "You are a helpful assistant. Don't use emojis, markdown and special characters. Please answer the user's words in a short and moderate manner. All answers should be in Korean.",
)
try:
    with narubot.STT(config) as stt:
        stt.process_audio_loop()
except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"An error occurred: {e}")