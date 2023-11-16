import uuid
from typing import Any

from deepsearchai.enums import MEDIA_TYPE
from deepsearchai.sources.data_source import DataSource
from .base import BaseEmbeddingModel


class Whisper(BaseEmbeddingModel):
    MODEL_NAME = "base"
    SUPPORTED_MEDIA_TYPES = [MEDIA_TYPE.AUDIO, MEDIA_TYPE.VIDEO]

    def __init__(self):
        self.model = None

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
        self._load_model()
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
            ids.append(str(uuid.uuid4()))

        result = {"documents": documents, "metadata": metadata, "ids": ids}
        return result

    def get_text_encoding(self, query: str):
        return {"text": query}

    def get_collection_name(self, media_type: MEDIA_TYPE):
        return "deepsearch-{}".format(media_type.name.lower())

    def _load_model(self):
        if not self.model:
            try:
                import whisper

                # Load the Whisper Model
                self.model = whisper.load_model(self.MODEL_NAME)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "The required dependencies for audio/video are not installed."
                    ' Please install with `pip install --upgrade "deepsearchai[audio]"` '
                    'or `pip install --upgrade "deepsearchai[video]"`'
                )
