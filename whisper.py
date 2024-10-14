import os
os.environ['OLLAMA_HOST'] = 'localhost:11434'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pyaudio
import numpy as np
import wave
import ollama
import pyttsx3
from faster_whisper import WhisperModel

# 모델 설정
WHISPER = "medium"
OLLAMA = "gemma2:27b"
PROMPT_SYSTEM = "You're a Nintendo Famicom. Don't use emojis, markdown and special characters. Please answer the user's words in a short and moderate manner. All answers should be in Korean."

# 마이크 입력 설정
CHUNK = 1024  # 오디오 청크 크기
FORMAT = pyaudio.paInt16  # 16비트 오디오 포맷
CHANNELS = 1  # 모노 채널
RATE = 16000  # 샘플링 레이트 (Whisper에 맞춰 16kHz)
THRESHOLD = 500  # 음성 시작/종료를 감지할 볼륨 임계값
SILENCE_LIMIT = 1  # 음성이 없을 때 대화 종료로 간주할 시간(초)

# TTS 출력 설정
TTS_RATE = 200  # TTS 음성 속도
TTS_VOLUME = 1.0  # TTS 볼륨

# 초기화
llm = WhisperModel(WHISPER, device="cpu", compute_type="float32")
portaudio = pyaudio.PyAudio()
tts = pyttsx3.init()

# 마이크에서 오디오 스트림 열기
microphone = portaudio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK)

# 오디오 데이터를 저장할 배열
audio_frames = []
silent_chunks = 0
speaking = False

# ollama 설정
if len(PROMPT_SYSTEM) > 0:
    messages = [{"role": "system", "content": PROMPT_SYSTEM}]
    max_messages = 33
    pop_at = 1
else:
    messages = []
    max_messages = 32
    pop_at = 0

# tts 엔진 설정
tts.setProperty('rate', TTS_RATE)
tts.setProperty('volume', TTS_VOLUME)
wav_file = "conversation.wav"

def is_silent(data):
    """오디오 데이터의 볼륨이 임계값 이하인지를 판단"""
    return np.abs(np.frombuffer(data, dtype=np.int16)).mean() < THRESHOLD

def save_wav(filename, frames):
    """녹음된 오디오를 WAV 파일로 저장"""
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(portaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def speech_to_text(file_path):
    """Whisper API를 호출하여 오디오 파일을 텍스트로 변환"""
    audio_file = open(file_path, "rb")
    segments, _ = llm.transcribe(audio_file, language="ko")
    segments = list(segments)
    return str.join(" ", [segment.text.strip() for segment in segments])

def chat_with_ollama(text):
    msg = {"role": "user", "content": text}
    messages.append(msg)
    response = ollama.chat(
        model=OLLAMA,
        messages=messages,
        stream=False,
    )
    msg = response['message']
    messages.append(msg)
    if len(messages) > max_messages:
        messages.pop(pop_at)
        messages.pop(pop_at)
    return msg['content']

def text_to_speech(text):
    tts.say(text)
    tts.runAndWait()

def flush_stream():
    """마이크 입력 버퍼 초기화"""
    global audio_frames, silent_chunks
    audio_frames = []
    silent_chunks = 0
    print("Listening...")

# 실시간 오디오 처리 루프
flush_stream()
while True:
    data = microphone.read(CHUNK)
    if is_silent(data):
        silent_chunks += 1
        if speaking and silent_chunks > SILENCE_LIMIT * RATE / CHUNK:
            print("Processing...")
            speaking = False
            save_wav(wav_file, audio_frames)
            text = speech_to_text(wav_file)
            response = chat_with_ollama(text)
            text_to_speech(response)
            flush_stream()
    else:
        silent_chunks = 0
        if not speaking:
            speaking = True
        audio_frames.append(data)
