import numpy as np
import sounddevice as sd
import webrtcvad
import asyncio
from queue import Queue
from threading import Thread
from termcolor import colored
import openai
import os
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor
import threading
import json
from datetime import datetime
import aiofiles  # For async file operations
import re
import pygame
import tempfile
from collections import deque
import asyncio.queues
from functools import partial
from shutil import get_terminal_size
import math
import requests
from huggingface_hub import InferenceClient
from io import BytesIO
import wave

# Initialize Groq client
groq_client = openai.AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

openai_client = openai.AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize WebRTC VAD
webrtc_vad = webrtcvad.Vad(3)

# Optimize audio settings
SAMPLE_RATE = 16000
FRAME_DURATION = 20  # Reduced from 30 to 20ms for faster processing
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
SILENCE_THRESHOLD = 5  # Reduced from 10 to 5 for faster response

# Add buffer size limit to prevent processing very long audio segments
MAX_BUFFER_SIZE = SAMPLE_RATE * 10  # Max 10 seconds of audio

# Global variables
audio_buffer = []
is_speaking = False
silence_frames = 0
thread_safe_queue = Queue()  # Regular Queue for thread safety
chat_history = []
response_interrupt_event = asyncio.Event()  # Add this for interruption control
is_processing_response = False  # Flag to track if we're currently processing a response

# Add after other global variables
CHAT_DIRECTORY = "chat_logs"
CHAT_FILE = f"{CHAT_DIRECTORY}/chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Add these constants at the top with other constants
SYSTEM_PROMPT = """You are a voice-enabled AI assistant. Keep responses brief and conversational - aim for 1-2 short sentences. 
When users ask if you can hear them, simply say "Yes, I can understand you through voice transcription. How can I help?"
Be natural and friendly."""

# Update the chat history initialization
chat_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# Add these constants after other constants
VOICE_METER_WIDTH = 30  # Width of the voice level meter
VOICE_METER_THRESHOLD = 300  # Increased from 100 to 300
VAD_AGGRESSIVENESS = 3  # Increased from 2 to 3 (maximum aggressiveness)
ENERGY_THRESHOLD = 100  # New threshold for speech detection

# Add this function after other utility functions
def draw_voice_meter(energy_level):
    """Draw a real-time voice level meter in the terminal."""
    terminal_width = get_terminal_size().columns
    meter_width = min(VOICE_METER_WIDTH, terminal_width - 20)
    
    # Normalize energy level to 0-1 range
    normalized_energy = min(1.0, max(0.0, math.log10(energy_level + 1) / 4.0))
    filled_bars = int(normalized_energy * meter_width)
    
    # Create the meter
    meter = '█' * filled_bars + '░' * (meter_width - filled_bars)
    
    # Color based on level
    if normalized_energy > 0.7:
        color = "red"
    elif normalized_energy > 0.3:
        color = "green"
    else:
        color = "blue"
    
    print(colored(f"\r🎤 Level: |{meter}|", color), end='', flush=True)

# Add this function for chat saving
async def save_chat_history():
    try:
        # Create directory if it doesn't exist
        os.makedirs(CHAT_DIRECTORY, exist_ok=True)
        
        chat_data = {
            "timestamp": datetime.now().isoformat(),
            "conversation": chat_history
        }
        
        async with aiofiles.open(CHAT_FILE, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(chat_data, indent=2, ensure_ascii=False))
            
        print(colored("💾 Chat saved", "blue"))
    except Exception as e:
        print(colored(f"❌ Error saving chat: {str(e)}", "red"))

# Add these after other global variables
audio_queue = asyncio.queues.Queue()  # Queue for audio files to play
current_audio = None
pygame.mixer.init(frequency=24000)  # Initialize pygame mixer

def detect_sentence_end(text):
    """Detect if text ends with sentence-ending punctuation."""
    return bool(re.search(r'[.!?]\s*$', text))

class AudioManager:
    def __init__(self):
        self.current_sound = None
        self.playing = False
        self._channel = None  # Add channel tracking
    
    def play_audio(self, audio_file):
        try:
            if self.current_sound and pygame.mixer.get_busy():
                pygame.mixer.stop()
            self.current_sound = pygame.mixer.Sound(audio_file)
            self._channel = self.current_sound.play()  # Store the channel
            self.playing = True
        except Exception as e:
            print(colored(f"❌ Audio playback error: {str(e)}", "red"))
    
    def stop_audio(self):
        if self.playing:
            if self._channel and self._channel.get_busy():
                self._channel.stop()  # Stop the specific channel
            pygame.mixer.stop()  # Backup stop all
            self.playing = False
            self.current_sound = None
            self._channel = None

audio_manager = AudioManager()

class ConversationState:
    def __init__(self):
        self.is_speaking = False
        self.is_processing_response = False
        self.response_interrupt_event = asyncio.Event()
        self.audio_queue = asyncio.queues.Queue()
        self.should_stop_audio = asyncio.Event()
        self.current_audio_task = None
        self.audio_tasks = set()
    
    async def interrupt(self):
        """Interrupt all ongoing processes"""
        try:
            print(colored("\n⚡ Interrupting current turn...", "yellow"))
            self.response_interrupt_event.set()
            self.should_stop_audio.set()
            
            # Immediate audio stop
            audio_manager.stop_audio()
            
            # Cancel tasks without awaiting them
            tasks_to_cancel = self.audio_tasks.copy()
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()
            
            # Clear the set
            self.audio_tasks.clear()
            
            # Clear audio queue without awaiting
            try:
                while True:
                    audio_file = self.audio_queue.get_nowait()
                    if isinstance(audio_file, str) and os.path.exists(audio_file):
                        os.unlink(audio_file)
            except asyncio.QueueEmpty:
                pass
            
            return True
        except Exception as e:
            print(colored(f"❌ Interrupt error in async: {str(e)}", "red"))
            return False

    def reset(self):
        """Reset state for next turn"""
        self.response_interrupt_event.clear()
        self.should_stop_audio.clear()
        self.is_processing_response = False
        self.current_audio_task = None
        self.audio_tasks.clear()

class ThreadSafeConversationState(ConversationState):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
    
    def interrupt_from_callback(self):
        """Thread-safe version of interrupt for use in callback"""
        if self.loop and self.loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self.interrupt(), self.loop)
                # Don't await the future.result() directly
                return True
            except Exception as e:
                print(colored(f"❌ Interrupt error in callback: {str(e)}", "red"))
                return False

# Add this at the global scope instead
conversation_state = None  # Will be initialized in main_async

# Add these constants after other API clients
HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY)

# Add fallback endpoints
ENDPOINTS = {
    "whisper": {
        "primary": "groq",
        "fallback": "openai/whisper-large-v3"
    },
    "tts": {
        "primary": "openai",
        "fallback": "microsoft/speecht5_tts"
    },
    "chat": {
        "primary": "groq",
        "fallback": "mistralai/mixtral-8x7b-instruct-v0.1"
    }
}

# Add after ENDPOINTS definition
def service_indicator(service, operation):
    icons = {
        "groq": "⚡",
        "openai": "🔵",
        "huggingface": "🤗"
    }
    return f"{icons.get(service, '❓')} Using {service.title()} for {operation}"

# Update transcribe_audio function to include fallback
async def transcribe_audio(audio_data):
    try:
        # Try primary (Groq) first
        response = await try_groq_transcription(audio_data)
        if response:
            return response
            
        # Fallback to Hugging Face if primary fails
        print(colored("⚠️ Falling back to Hugging Face for transcription", "yellow"))
        response = await try_hf_transcription(audio_data)
        return response

    except Exception as e:
        print(colored(f"❌ Transcription Error: {str(e)}", "red"))
        return None

async def try_groq_transcription(audio_data):
    """Primary transcription using Groq"""
    try:
        print(colored(service_indicator("groq", "transcription"), "green"))
        # If we're processing a response, interrupt it
        if conversation_state.is_processing_response:
            conversation_state.response_interrupt_event.set()
            await asyncio.sleep(0.1)
        
        audio_array = np.array(audio_data, dtype=np.int16)
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_array.tobytes())
        
        wav_buffer.seek(0)
        response = await groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=("audio.wav", wav_buffer),
            response_format="text"
        )
        
        if response:
            print(colored(f"\n🎤 You: {response}", "cyan"))
            await get_chat_response(response)
            
        return response
    except Exception as e:
        print(colored(f"❌ Groq Transcription Error: {str(e)}", "red"))
        return None

async def try_hf_transcription(audio_data):
    """Fallback transcription using Hugging Face"""
    try:
        print(colored(service_indicator("huggingface", "transcription"), "yellow"))
        audio_array = np.array(audio_data, dtype=np.int16)
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_array.tobytes())
        
        wav_buffer.seek(0)
        response = await asyncio.to_thread(
            hf_client.audio_to_text,
            data=wav_buffer,
            model=ENDPOINTS["whisper"]["fallback"]
        )
        return response["text"]
    except Exception as e:
        print(colored(f"❌ HF Transcription Error: {str(e)}", "red"))
        return None

# Add after the transcription functions but before generate_and_play_audio
async def try_openai_tts(text):
    """Primary TTS using OpenAI"""
    try:
        print(colored(service_indicator("openai", "TTS"), "blue"))
        if conversation_state.should_stop_audio.is_set():
            return None
        
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        if conversation_state.should_stop_audio.is_set():
            return None
            
        return response.content
    except Exception as e:
        print(colored(f"❌ OpenAI TTS Error: {str(e)}", "red"))
        return None

async def try_hf_tts(text):
    """Fallback TTS using Hugging Face"""
    try:
        print(colored(service_indicator("huggingface", "TTS"), "yellow"))
        response = await asyncio.to_thread(
            hf_client.text_to_speech,
            text=text,
            model=ENDPOINTS["tts"]["fallback"]
        )
        return response
    except Exception as e:
        print(colored(f"❌ HF TTS Error: {str(e)}", "red"))
        return None

# Similar fallback pattern for TTS
async def generate_and_play_audio(text, sentence_index):
    try:
        # Process sentences in order
        response = await try_openai_tts(text)
        if not response and HUGGINGFACE_API_KEY:
            print(colored("⚠️ Falling back to Hugging Face for TTS", "yellow"))
            response = await try_hf_tts(text)
            
        if response:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tmp_file.write(response)
                audio_file = tmp_file.name
            
            # Wait for previous sentences to complete
            while sentence_index > 0 and not conversation_state.should_stop_audio.is_set():
                if pygame.mixer.get_busy():
                    await asyncio.sleep(0.05)
                else:
                    break
                    
            await conversation_state.audio_queue.put((sentence_index, audio_file))
            print(colored(f"🔊 Generated audio for sentence {sentence_index}: {text[:30]}...", "blue"))
            
    except Exception as e:
        print(colored(f"❌ TTS Error: {str(e)}", "red"))

async def audio_player():
    """Continuously play audio files from the queue."""
    pending_audio = {}  # Dictionary to store out-of-order audio files
    next_index = 0  # Track the next expected sentence index
    
    while True:
        try:
            if conversation_state.should_stop_audio.is_set():
                pending_audio.clear()
                next_index = 0
                await asyncio.sleep(0.1)
                continue

            if not conversation_state.audio_queue.empty():
                index, audio_file = await conversation_state.audio_queue.get()
                
                if conversation_state.should_stop_audio.is_set():
                    if audio_file and os.path.exists(audio_file):
                        os.unlink(audio_file)
                    continue

                # Store out-of-order audio files
                if index != next_index:
                    pending_audio[index] = audio_file
                    continue
                
                # Play the current audio file
                audio_manager.play_audio(audio_file)
                
                # Wait for audio to finish or interruption
                while not conversation_state.should_stop_audio.is_set():
                    if not pygame.mixer.get_busy():
                        break
                    await asyncio.sleep(0.05)
                
                # Clean up current file
                if audio_file and os.path.exists(audio_file):
                    os.unlink(audio_file)
                
                next_index += 1
                
                # Check for pending audio files that can now be played
                while next_index in pending_audio and not conversation_state.should_stop_audio.is_set():
                    audio_file = pending_audio.pop(next_index)
                    audio_manager.play_audio(audio_file)
                    
                    while not conversation_state.should_stop_audio.is_set():
                        if not pygame.mixer.get_busy():
                            break
                        await asyncio.sleep(0.05)
                    
                    if audio_file and os.path.exists(audio_file):
                        os.unlink(audio_file)
                    next_index += 1
                
            await asyncio.sleep(0.1)
        except Exception as e:
            print(colored(f"❌ Audio player error: {str(e)}", "red"))

async def get_chat_response(user_message):
    try:
        if conversation_state.response_interrupt_event.is_set():
            return None
            
        chat_history.append({"role": "user", "content": user_message})
        await save_chat_history()
        print(colored("\n🤖 Getting AI response...", "yellow"))
        
        is_processing_response = True
        response_stream = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_history,
            stream=True,
            temperature=0.7,
        )
        
        full_response = ""
        current_sentence = ""
        sentence_index = 0
        
        async for chunk in response_stream:
            if conversation_state.response_interrupt_event.is_set():
                print(colored("\n⚡ Response interrupted by user", "yellow"))
                break
                
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                current_sentence += content
                print(colored(content, "green"), end="", flush=True)
                
                # Check for sentence completion
                if detect_sentence_end(current_sentence):
                    if not conversation_state.response_interrupt_event.is_set():
                        audio_task = asyncio.create_task(
                            generate_and_play_audio(current_sentence.strip(), sentence_index)
                        )
                        conversation_state.audio_tasks.add(audio_task)
                        audio_task.add_done_callback(
                            lambda t: conversation_state.audio_tasks.discard(t)
                        )
                        sentence_index += 1
                    current_sentence = ""
        
        # Handle any remaining text
        if current_sentence.strip() and not conversation_state.response_interrupt_event.is_set():
            audio_task = asyncio.create_task(
                generate_and_play_audio(current_sentence.strip(), sentence_index)
            )
            conversation_state.audio_tasks.add(audio_task)
            audio_task.add_done_callback(
                lambda t: conversation_state.audio_tasks.discard(t)
            )
        
        if full_response:
            print()  # New line after streaming response
            chat_history.append({"role": "assistant", "content": full_response})
            await save_chat_history()
            
        return full_response
    except Exception as e:
        print(colored(f"❌ Chat Error: {str(e)}", "red"))
        return None
    finally:
        conversation_state.is_processing_response = False
        conversation_state.response_interrupt_event.clear()

async def process_thread_queue():
    while True:
        try:
            if not thread_safe_queue.empty():
                audio_data = thread_safe_queue.get_nowait()
                asyncio.create_task(transcribe_audio(audio_data))  # Don't await, process concurrently
            await asyncio.sleep(0.01)  # Reduced sleep time
        except Exception as e:
            print(colored(f"Error: {str(e)}", "red"))

def record_callback(indata, frames, time, status):
    global audio_buffer, silence_frames
    
    audio_data = (indata[:, 0] * 32768).astype(np.int16)
    frame_energy = np.mean(np.abs(audio_data))
    
    # More stringent speech detection
    is_speech = (webrtc_vad.is_speech(audio_data.tobytes(), sample_rate=SAMPLE_RATE) and 
                frame_energy > ENERGY_THRESHOLD)
    
    # Draw voice meter if energy is above threshold
    if frame_energy > VOICE_METER_THRESHOLD:
        draw_voice_meter(frame_energy)
    
    if is_speech:
        # Add consecutive speech frame counter
        if not hasattr(record_callback, 'speech_frames'):
            record_callback.speech_frames = 0
        record_callback.speech_frames += 1
        
        # Only trigger voice detection after multiple consecutive speech frames
        if record_callback.speech_frames >= 3:  # Require 3 consecutive speech frames
            if not conversation_state.is_speaking:
                conversation_state.interrupt_from_callback()
                print(colored("\n🎙️ Voice detected", "blue"))
                conversation_state.is_speaking = True
            silence_frames = 0
            if len(audio_buffer) < MAX_BUFFER_SIZE:
                audio_buffer.extend(audio_data.tolist())
    else:
        if hasattr(record_callback, 'speech_frames'):
            record_callback.speech_frames = 0
        
        if conversation_state.is_speaking:
            silence_frames += 1
            if silence_frames > SILENCE_THRESHOLD:
                if len(audio_buffer) > FRAME_SIZE * 3:
                    print(colored("Processing", "blue"))
                    thread_safe_queue.put(audio_buffer.copy())
                conversation_state.is_speaking = False
                conversation_state.reset()
                audio_buffer = []
            elif len(audio_buffer) < MAX_BUFFER_SIZE:
                audio_buffer.extend(audio_data.tolist())

async def main_async():
    # Store the event loop
    loop = asyncio.get_running_loop()
    
    # Create conversation state with loop
    global conversation_state
    conversation_state = ThreadSafeConversationState(loop)
    
    os.makedirs(CHAT_DIRECTORY, exist_ok=True)
    print(colored(f"📁 Saving chat to: {CHAT_FILE}", "blue"))
    
    # Initialize pygame mixer here instead of global scope
    pygame.mixer.init(frequency=24000)
    
    process_task = asyncio.create_task(process_thread_queue())
    audio_task = asyncio.create_task(audio_player())
    
    print(colored("🎧 Listening... (Press Ctrl+C to stop)", "cyan"))
    
    stream = sd.InputStream(
        callback=record_callback,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype=np.float32,
        latency='low'
    )
    
    with stream:
        try:
            while True:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print(colored("\n👋 Stopping audio capture...", "yellow"))
        finally:
            process_task.cancel()
            audio_task.cancel()
            audio_manager.stop_audio()
            pygame.mixer.quit()  # Clean up pygame
            try:
                await asyncio.gather(process_task, audio_task)
            except asyncio.CancelledError:
                pass

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()