Real-time Voice Chat Application with AI
======================================

This application creates a real-time voice chat interface using advanced AI technologies. Here's how it works:

Key Components:
-------------

1. Voice Activity Detection (VAD)
- Uses WebRTC VAD to detect when the user is speaking
- Automatically starts/stops audio capture based on voice activity
- Configurable silence threshold for better response times

2. Speech-to-Text (Using Groq Whisper)
- Transcribes user's speech in real-time using Whisper Large v3 Turbo
- Optimized for fast, accurate transcriptions
- Uses BytesIO for efficient audio processing

3. AI Chat Response (Using GPT-4o-mini)
- Processes transcribed text through GPT-4o-mini
- Streams responses for immediate feedback
- Maintains conversation history for context

4. Text-to-Speech (Using OpenAI TTS)
- Converts AI responses to natural-sounding speech
- Uses sentence detection for smooth audio playback
- Handles audio generation and playback asynchronously

Real-time Flow:
-------------

1. When you speak:
   - VAD detects your voice
   - Audio is captured and buffered
   - Any playing audio is immediately stopped

2. When you pause:
   - Audio is sent for transcription
   - Transcribed text is displayed
   - AI generates a streaming response

3. As the AI responds:
   - Text appears in real-time
   - Speech is generated per sentence
   - Audio plays while next sentences process

4. Interrupt Handling:
   - Speaking during AI response stops playback
   - System immediately switches to listening mode
   - Previous operations are cleanly terminated

Technical Features:
-----------------

- Asynchronous processing for optimal performance
- Thread-safe queue system for audio handling
- Efficient memory management with buffer limits
- Colored console output for status tracking
- Automatic chat history saving
- Configurable audio parameters

Requirements:
-----------
- Python packages: numpy, sounddevice, webrtcvad, asyncio, termcolor, openai, aiofiles, pygame
- API keys for Groq and OpenAI services
- System audio input/output capabilities

The system is designed for minimal latency and natural conversation flow, making it feel like talking to a real person while leveraging powerful AI capabilities.
