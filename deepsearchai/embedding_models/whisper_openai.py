import io
import os
import uuid
from typing import Any

import openai
from pydub import AudioSegment

from deepsearchai.enums import MEDIA_TYPE
from deepsearchai.sources.data_source import DataSource
from .base import BaseEmbeddingModel


# This is a model to transcribe audio to text using openai APIs, hence user needs to have the approrpiate env variables
# set to be able to use it
class WhisperOpenAi(BaseEmbeddingModel):
    MODEL_NAME = "whisper-1"
    SUPPORTED_MEDIA_TYPES = [MEDIA_TYPE.AUDIO, MEDIA_TYPE.VIDEO]

    def __init__(self):
        pass

    def get_media_encoding(
        self, file: str, data_type: MEDIA_TYPE, datasource: DataSource
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
        audio = AudioSegment.from_file(file)
        chunk_size = self._get_chunk_size(audio.duration_seconds)
        # Split the AudioSegment file into chunks
        chunks = audio[::chunk_size]
        documents = []
        metadata = []
        ids = []
        for i, chunk in enumerate(chunks):
            # Transcribe the audio chunk
            buffer = io.BytesIO()
            buffer.name = ".tmp/deepsearch/youtube/tmp.wav"
            chunk.export(buffer, format="wav")

            transcript = openai.Audio.transcribe(self.MODEL_NAME, buffer)
            start = i * chunk_size
            end = start + chunk.duration_seconds
            documents.append(transcript.get("text"))
            metadata.append(
                {
                    "start": start,
                    "end": end,
                }
            )
            ids.append(str(uuid.uuid4()))

        if datasource != DataSource.LOCAL:
            self._delete_file(file)
        result = {"documents": documents, "metadata": metadata, "ids": ids}
        return result

    def get_text_encoding(self, query: str):
        return {"text": query}

    def get_collection_name(self, media_type: MEDIA_TYPE):
        return "deepsearch-{}".format(media_type.name.lower())

    def _get_chunk_size(self, total_duration: float):
        # Hardcoded chunk size of 5 minutes, but can have a smarter approach based on the total length of the audio
        return 300000

    def _delete_file(self, filename):
        os.remove(filename)
