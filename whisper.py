"""
whisper.py - A script for recording and processing audio using Whisper, Ollama, and TTS engines.

This script allows real-time interaction by transcribing audio from a microphone input into text using the Whisper model, 
conversing with an Ollama chatbot based on the transcribed text, and converting responses back to speech using pyttsx3.

The script initializes PyAudio for capturing audio data from a microphone, uses Faster-Whisper for transcription, 
interacts with Ollama via API calls, and utilizes pyttsx3 for text-to-speech conversion. It processes the audio in real-time, 
saving chunks of audio when speech is detected and transcribing them to text before interacting with the chatbot.
"""

import os
os.environ['OLLAMA_HOST'] = 'localhost:11434'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pyaudio
import numpy as np
import wave
import ollama
import pyttsx3
from faster_whisper import WhisperModel

# Configuration settings for the script.
WHISPER = "medium"  # The model size to use for transcription.
OLLAMA = "gemma2:27b"  # The Ollama model to use for chatbot interactions.
PROMPT_SYSTEM = "You're a Nintendo Famicom. Don't use emojis, markdown and special characters. Please answer the user's words in a short and moderate manner. All answers should be in Korean."  # System prompt for Ollama.
WAV_FILE = "conversation.wav"  # File to save audio data as WAV.
CHUNK = 1024  # Number of frames per buffer for PyAudio.
FORMAT = pyaudio.paInt16  # Audio format for recording.
CHANNELS = 1  # Number of channels for recording.
RATE = 16000  # Sampling rate for recording.
THRESHOLD = 500  # Threshold to detect silence in audio data.
SILENCE_LIMIT = 1  # Maximum number of silent chunks before considering speech as ended.
TTS_RATE = 200  # Speed of text-to-speech output.
TTS_VOLUME = 1.0  # Volume level for text-to-speech output.

def initialize_whisper() -> WhisperModel:
    """
    Initialize and return a WhisperModel instance.
    
    This function sets up and returns a WhisperModel object using the specified parameters.
    The model is initialized on "cpu" with a compute type of "float32".
    
    Returns:
        WhisperModel: An instance of the WhisperModel class configured as specified.
    """
    return WhisperModel(WHISPER, device="cpu", compute_type="float32")

def initialize_pyaudio() -> pyaudio.PyAudio:
    """Initialize and return a PyAudio instance.
    
    This function sets up and returns a PyAudio object, which is used for audio processing in Python.
    
    Returns:
        pyaudio.PyAudio: An instance of the PyAudio class, initialized and ready to use for audio operations.
    """
    return pyaudio.PyAudio()

def initialize_tts():
    """
    Initializes and returns a pyttsx3 engine instance with specified properties:
    
    - 'rate': The speed at which the text-to-speech engine reads the text, in words per minute.
    - 'volume': The volume level for the speech output, ranging from 0.0 to 1.0.
    
    Returns:
        pyttsx3.engine.Engine: An instance of the pyttsx3 TTS engine with the specified properties set.
    """
    tts = pyttsx3.init()
    tts.setProperty('rate', TTS_RATE)
    tts.setProperty('volume', TTS_VOLUME)
    return tts

def initialize_microphone(portaudio: pyaudio.PyAudio) -> pyaudio.Stream:
    """Open a microphone input stream using PyAudio.

    This function configures and opens an audio stream that captures audio input 
    from a microphone connected to the system via PortAudio. The stream is set up 
    with the following parameters:
    
    - format: Sample format, here set to 16-bit PCM (pyaudio.paInt16).
    - channels: Number of audio channels, configured for mono input (CHANNELS = 1).
    - rate: Sampling rate, set to 44100 Hz which is standard CD quality.
    - input: True indicates that this stream is for capturing audio input from the microphone.
    - frames_per_buffer: The size of the buffer used to read chunks of audio data (CHUNK = 1024 samples).
    
    Returns:
        pyaudio.Stream: An open stream object which can be used to capture audio data.
    """
    return portaudio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK)

def initialize_ollama():
    """
    Initializes the Ollama chatbot by setting up the initial messages and parameters based on the system prompt.

    This function checks if there is a system prompt defined in PROMPT_SYSTEM. If it exists, it creates an initial message 
    with the role 'system' and the content set to PROMPT_SYSTEM. It also sets default values for max_messages and pop_at 
    based on whether the system prompt is present or not.

    Returns:
        tuple: A tuple containing three elements:
            - messages (list): A list of dictionaries representing chatbot messages, where each dictionary contains a 'role' 
                                and 'content'. If no system prompt is defined, this will be an empty list.
            - max_messages (int): The maximum number of messages that can be stored in the conversation history before any are removed.
                                    This defaults to 33 if a system prompt is present, otherwise it's set to 32.
            - pop_at (int): The index at which the oldest message should be removed when the max_messages limit is reached.
                            If no system prompt is defined, this value is set to 0.
    """
    if len(PROMPT_SYSTEM) > 0:
        messages = [{"role": "system", "content": PROMPT_SYSTEM}]
        max_messages = 33
        pop_at = 1
    else:
        messages = []
        max_messages = 32
        pop_at = 0
    return messages, max_messages, pop_at

# Initialize all components.
llm = initialize_whisper()
portaudio = initialize_pyaudio()
tts = initialize_tts()
microphone = initialize_microphone(portaudio)
messages, max_messages, pop_at = initialize_ollama()

# Global variables for processing audio data.
audio_frames = []
silent_chunks = 0
speaking = False

def is_silent(data: bytes) -> bool:
    """Check if the given audio data chunk contains only silence.
    
    This function analyzes a byte string representing an audio chunk and determines
    whether it is considered silent based on the average absolute value of its samples
    compared to a predefined threshold. The threshold can be adjusted according to the 
    specific characteristics of your audio data, such as volume level or quality settings.
    
    Args:
        data (bytes): A byte string representing an audio chunk.
        
    Returns:
        bool: True if the average absolute value of the samples is below the threshold,
              False otherwise.
    """
    # Convert bytes to numpy array and calculate mean of absolute values
    return np.abs(np.frombuffer(data, dtype=np.int16)).mean() < THRESHOLD

def save_wav(filename, frames):
    """Save the list of audio frames to a WAV file.

    This function takes a filename and a list of audio frames as input. It opens a new WAV file with the given filename, sets its parameters (number of channels, sample width, frame rate), and writes the provided frames into it. Finally, it closes the file.

    Args:
        filename (str): The name of the file to which the audio frames will be saved.
        frames (list of bytes): A list containing the raw byte data for each audio frame.

    Returns:
        None

    Example:
        To save a WAV file from a list of frames, you would call this function like so:
        
        >>> frames = [...]  # Your list of audio frames
        >>> save_wav('output.wav', frames)
    """
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(portaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def speech_to_text(file_path:str):
    """Transcribe the audio file using the Whisper model and return the text.
    
    This function takes an audio file path as input, opens the file in binary mode,
    and uses a pre-trained version of the Whisper model from the llm library to 
    transcribe the audio into segments. Each segment's text is stripped of leading
    and trailing whitespace before being joined together with spaces to form a single string.
    
    Args:
        file_path (str): The path to the audio file that needs to be transcribed.
        
    Returns:
        str: A concatenated string containing all the transcribed text from the audio segments, 
             with each segment's text separated by spaces and leading/trailing whitespace removed.
    
    Example:
        To transcribe an audio file named 'audio_sample.wav', you would call:
        
        >>> speech_to_text('path/to/audio_sample.wav')
    """
    audio_file = open(file_path, "rb")
    segments, _ = llm.transcribe(audio_file, language="ko")
    segments = list(segments)
    return str.join(" ", [segment.text.strip() for segment in segments])

def chat_with_ollama(text:str):
    """
    Sends a user's text to the Ollama chatbot and retrieves a response.

    Args:
        text (str): The input text that the user wants to send to the chatbot.
        
    Returns:
        str: The content of the bot's response message.
        
    This function performs the following steps:
    1. Appends the user's text as a new message to the `messages` list with the role 'user'.
    2. Sends the complete messages history to the Ollama chatbot using the `ollama.chat` method.
    3. Appends the bot's response message to the `messages` list.
    4. If the number of messages exceeds a predefined maximum (`max_messages`), it removes the oldest message (as specified by `pop_at`) from the beginning of the messages list.
    """
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
    """
    Convert the given text to speech and play it using the installed TTS (Text-to-Speech) engine.
    
    Args:
        text (str): The input text that needs to be converted to speech.
        
    Returns:
        None
    """
    tts.say(text)
    tts.runAndWait()

def flush_stream():
    """
    Resets the global `audio_frames` list and the `silent_chunks` counter.
    
    This function sets `audio_frames` to an empty list and resets the `silent_chunks` variable to zero, which is used to keep track of consecutive silent chunks in audio data. The message "Listening..." is also printed to indicate that the system is ready to listen again.
    
    Parameters:
        None
        
    Returns:
        None
    """
    global audio_frames, silent_chunks
    audio_frames = []
    silent_chunks = 0
    print("Listening...")

def process_audio():
    """Process the audio stream from the microphone in real-time."""
    global audio_frames, silent_chunks, speaking
    flush_stream()
    
    while True:
        data = microphone.read(CHUNK)
        if is_silent(data):
            silent_chunks += 1
            if speaking and silent_chunks > SILENCE_LIMIT * RATE / CHUNK:
                print("Processing...")
                speaking = False
                save_wav(WAV_FILE, audio_frames)
                text = speech_to_text(WAV_FILE)
                response = chat_with_ollama(text)
                text_to_speech(response)
                flush_stream()
        else:
            silent_chunks = 0
            if not speaking:
                speaking = True
            audio_frames.append(data)

# Start the main processing loop for audio data.
process_audio()