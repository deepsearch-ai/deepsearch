from typing import Any

import whisper

from ..enums import MEDIA_TYPE
from .base import BaseLLM


class Whisper(BaseLLM):
    SUPPORTED_MEDIA_TYPES = [MEDIA_TYPE.AUDIO]

    def __init__(self):
        # Create a Whisper recognizer.
        self.model = whisper.load_model("base")

    def get_media_encoding(self, data: Any, data_type: MEDIA_TYPE):
        """Get the media encoding using OpenAI's Whisper model.

        Args:
            data: The media data.
            dataType: The media type, either "audio" or "video".

        Returns:
            The media encoding, or None if the encoding could not be determined.
            :param data:
            :param data_type:
        """
        if data_type not in self.SUPPORTED_MEDIA_TYPES:
            raise ValueError("Unsupported dataType. Whisper model supports only {}".format(self.SUPPORTED_MEDIA_TYPES))
        transcription = self.model.transcribe(data)
        result = {
            "text": transcription.get("text"),
        }
        segments = []
        for segment in transcription.get("segments"):
            segments.append({
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text"),
            })

        result["segments"] = segments
        return result

    def get_text_encoding(self, query: str):
        return {
            "text": query
        }
