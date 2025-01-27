import narubot

config = narubot.Config(
    device = "cuda",
    stt_model = "medium",
    llm_model = "gemma2:27b",
    llm_system_prompt = "You're a Nintendo Famicom. Don't use emojis, markdown and special characters. Please answer the user's words in a short and moderate manner. All answers should be in Korean.",
)
try:
    with narubot.STT(config) as stt:
        stt.process_audio_loop(narubot.LLM(config), narubot.TTS(config))
except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"An error occurred: {e}")