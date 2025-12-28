import os
from typing import Optional

from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

_client: Optional[ElevenLabs] = None

def get_eleven_client() -> ElevenLabs:
    global _client
    if _client is None:
        if not ELEVENLABS_API_KEY:
            raise RuntimeError("ELEVENLABS_API_KEY is missing.")
        _client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    return _client

def text_to_speech_mp3_bytes(
        text: str,
        voice: Optional[str] = None,
        model: Optional[str] = None,
) -> bytes:
    """
    Returns MP3 bytes for the given text using ElevenLabs.
    """
    client = get_eleven_client()
    voice = voice or ELEVENLABS_VOICE_ID
    model = model or ELEVENLABS_MODEL_ID

    # The SDK returns an iterator of audio chunks for streaming
    audio_iter = client.text_to_speech.convert(
        text=text,
        voice_id=voice,
        model_id=model,
        output_format="mp3_44100_128",
    )

    return b"".join(audio_iter)