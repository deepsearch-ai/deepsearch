from typing import Any
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from .base import BaseEmbeddingModel
from ..enums import MEDIA_TYPE
from ..sources.data_source import DataSource


class BlipImageCaptioning(BaseEmbeddingModel):
    MODEL_NAME = "Salesforce/blip-image-captioning-base"

    def __init__(self):
        self.processor = BlipProcessor.from_pretrained(self.MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(self.MODEL_NAME)

    def get_media_encoding(self, data: Any, data_type: MEDIA_TYPE, datasource: DataSource):
        import pdb; pdb.set_trace()
        inputs = self.processor(data, return_tensors="pt")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return {"documents": caption, "meta_data": {}}

    def get_text_encoding(self, query: str):
        return {"text": query}

    def get_collection_name(self, media_type: MEDIA_TYPE):
        return "deepsearch-image-captioning"
