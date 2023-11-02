from typing import Any

import torch
import whisper

from ..enums import MEDIA_TYPE
from .base import BaseLLM


class Whisper(BaseLLM):
    MODEL_NAME = "clip-ViT-B-32"
    recognizer = None

    def __init__(self):
        # Create a Whisper recognizer.
        recognizer = whisper.Whisper(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def get_media_encoding(self, data: Any, dataType: MEDIA_TYPE):
        """Get the media encoding using OpenAI's Whisper model.

        Args:
            data: The media data.
            dataType: The media type, either "audio" or "video".

        Returns:
            The media encoding, or None if the encoding could not be determined.
        """
        global recognizer
        if dataType == "audio":
            recognizer.load_audio(data)
        elif dataType == "video":
            recognizer.load_video(data)
        else:
            raise ValueError("Invalid media type: {}".format(dataType))

        # Get the transcript from the recognizer.
        transcript = recognizer.transcribe()

        # Extract the media encoding from the transcript.
        media_encoding = None
        for line in transcript.split("\n"):
            if line.startswith("Encoding: "):
                media_encoding = line[9:].strip()
                break

        return media_encoding
