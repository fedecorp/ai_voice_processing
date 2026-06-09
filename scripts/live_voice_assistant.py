# ------------------------------------------------------------------------------------------
# AUTHOR: FEDERICO MARZULLO
# ------------------------------------------------------------------------------------------
# HOW TO RUN THIS CODE:
# 1) Make sure you have a .env file in the project root with your Gemini API key:
#       GOOGLE_API_KEY=AIza...your_key_here...
#    (Copy .env.example → .env and fill in your key. See Notebook 0 for details.)
# 2) From the project root, run this script:
# > python scripts\live_voice_assistant.py
# 3) Make sure you authorize any executions or microphone access permissions when prompted by your OS, antivirus or firewall connections
# 4) Close the script with Ctrl+C in the terminal when you're done testing the live voice assistant

# ------------------------------------------------------------------------------------------
# ⚠️ A Critical Implementation Safety Note
# When running bidirectional streaming voice systems, always wear headphones. Because there is
# no software echo cancellation loop built directly into raw PyAudio byte arrays, any voice
# response coming out of your laptop's built-in speakers will feed directly back into your
# built-in microphone. This loop triggers the API's server-side interruption engine, causing
# the AI to continuously cut itself off mid-sentence.
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
import asyncio
import os
import sys
import pyaudio
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types

load_dotenv(find_dotenv())

# 1. Hardware Configuration Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
MIC_RATE = 16000       
SPEAKER_RATE = 24000   
CHUNK_SIZE = 512       

# 2. Model Selection
LIVE_MODEL = "gemini-3.1-flash-live-preview"

async def send_microphone_stream(session, mic_stream):
    """Continuously captures microphone audio chunks and pipes them to Gemini via WebSockets."""
    print("🎤 Microphone is live. Start speaking whenever you want...")
    while True:
        try:
            data = await asyncio.to_thread(
                mic_stream.read, CHUNK_SIZE, exception_on_overflow=False
            )
            if data:
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=data,
                        mime_type="audio/pcm;rate=16000"
                    )
                )
        except Exception as e:
            print(f"\n[Error recording/sending audio]: {e}")
            break
        await asyncio.sleep(0.001)

async def receive_speaker_stream(session, speaker_stream):
    """Listens for the model's live response chunks across multiple conversational turns."""
    print("🤖 Assistant is listening and ready to talk...")
    
    # FIX: Wrapped in a continuous loop because session.receive() breaks when a turn finishes
    while True:
        try:
            async for response in session.receive():
                server_content = response.server_content
                
                if server_content:
                    # Print text transcription live as the audio plays
                    if server_content.output_transcription:
                        print(server_content.output_transcription.text, end="", flush=True)
                    
                    # Playback audio generation handling
                    if server_content.model_turn:
                        for part in server_content.model_turn.parts:
                            if part.inline_data:
                                await asyncio.to_thread(
                                    speaker_stream.write, part.inline_data.data
                                )
                                
                # Line break on complete conversational turns
                if server_content and server_content.turn_complete:
                    print("\n")
                    
        except Exception as e:
            print(f"\n[Error receiving audio]: {e}")
            break
            
        # Small delay to prevent CPU thrashing when resetting the generator
        await asyncio.sleep(0.01)

async def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable is not set.")
        sys.exit(1)

    client = genai.Client()
    pya = pyaudio.PyAudio()

    mic_stream = pya.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=MIC_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    speaker_stream = pya.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SPEAKER_RATE,
        output=True,
        frames_per_buffer=CHUNK_SIZE
    )

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(
            parts=[types.Part(text="You are a real-time voice assistant. Keep your answers natural, concise, and dynamic.")]
        )
    )

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            print("⚡ Connection established with Gemini Live Engine.")
            
            await asyncio.gather(
                send_microphone_stream(session, mic_stream),
                receive_speaker_stream(session, speaker_stream)
            )
            
    except KeyboardInterrupt:
        print("\nStopping voice session...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        speaker_stream.stop_stream()
        speaker_stream.close()
        pya.terminate()
        print("🔌 Session closed cleanly.")

if __name__ == "__main__":
    asyncio.run(main())