import narubot

config = narubot.Config(
    device = "cuda",
    stt_start_commands=["헤이", "안녕", "시리야"],
    stt_stop_commands=["종료", "끝", "그만", "멈춰", "잘가"],
    stt_magic_commands=["시리", "패미컴", "네스"],
    stt_model = "large-v2",
    llm_model = "gemma2:27b",
    llm_system_prompt = "You're a Nintendo Famicom. Don't use emojis, markdown and special characters. Please answer the user's words in a short and moderate manner. All answers should be in Korean.",
)
try:
    with narubot.STT(config) as stt:
        stt.process_audio_loop()
except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"An error occurred: {e}")