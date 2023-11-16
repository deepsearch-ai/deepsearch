import uuid
from typing import Any

from deepsearchai.enums import MEDIA_TYPE
from deepsearchai.sources.data_source import DataSource
from .base import BaseEmbeddingModel


class BlipImageCaptioning(BaseEmbeddingModel):
    MODEL_NAME = "Salesforce/blip-image-captioning-base"

    def __init__(self):
        self.processor = None
        self.model = None

    def get_media_encoding(
        self, data: Any, data_type: MEDIA_TYPE, datasource: DataSource
    ):
        self._load_model()
        inputs = self.processor(data, return_tensors="pt")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        id = str(uuid.uuid4())
        return {"documents": [caption], "ids": [id], "metadata": [{"type": "caption"}]}

    def get_text_encoding(self, query: str):
        return {"text": query}

    def get_collection_name(self, media_type: MEDIA_TYPE):
        return "deepsearch-{}-captioning".format(media_type.name.lower())

    def _load_model(self):
        if not self.processor or not self.model:
            try:
                from transformers import (BlipForConditionalGeneration,
                                          BlipProcessor)

                self.processor = BlipProcessor.from_pretrained(self.MODEL_NAME)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.MODEL_NAME
                )
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "The required dependencies for audio/video are not installed."
                    ' Please install with `pip install --upgrade "deepsearchai[image]"`'
                )
