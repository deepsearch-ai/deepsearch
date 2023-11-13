import hashlib
from typing import Any

import whisper

from ..enums import MEDIA_TYPE
from ..sources.data_source import DataSource
from .base import BaseEmbeddingModel


class Whisper(BaseEmbeddingModel):
    MODEL_NAME = "base"
    SUPPORTED_MEDIA_TYPES = [MEDIA_TYPE.AUDIO, MEDIA_TYPE.VIDEO]

    def __init__(self):
        # Create a Whisper recognizer.
        self.model = whisper.load_model(self.MODEL_NAME)

    def get_media_encoding(
        self, data: Any, data_type: MEDIA_TYPE, datasource: DataSource
    ):
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
            raise ValueError(
                "Unsupported dataType. Whisper model supports only {}".format(
                    self.SUPPORTED_MEDIA_TYPES
                )
            )
        transcription = self.model.transcribe(data)
        documents = []
        metadata = []
        ids = []
        for segment in transcription.get("segments"):
            documents.append(segment.get("text"))
            metadata.append(
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                }
            )
            ids.append(hashlib.sha256((segment.get("text")).encode()).hexdigest())

        result = {"documents": documents, "metadata": metadata, "ids": ids}
        return result

    def get_text_encoding(self, query: str):
        return {"text": query}
