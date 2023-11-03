from typing import Any

from sentence_transformers import SentenceTransformer

from ..enums import MEDIA_TYPE
from .base import BaseLLM


class Clip(BaseLLM):
    MODEL_NAME = "clip-ViT-B-32"

    def __init__(self):
        self._load_model()

    def get_media_encoding(self, data: Any, dataType: MEDIA_TYPE):
        """
        Applies the CLIP model to evaluate the vector representation of the supplied image
        """
        if dataType != MEDIA_TYPE.IMAGE:
            raise ValueError("Unsupported dataType. Clip model supports only IMAGE")
        # try:
        #     # load image
        #     image = Image.open(url)
        # except FileNotFoundError:
        #     raise FileNotFoundError("The supplied file does not exist`")
        # except UnidentifiedImageError:
        #     raise UnidentifiedImageError("The supplied file is not an image`")

        image_features = self.model.encode(data)
        meta_data = {"media_type": "image"}
        return {"embedding": image_features.tolist(), "meta_data": meta_data}

    def get_text_encoding(self, query: str):
        """
        Applies the CLIP model to evaluate the vector representation of the supplied text
        """
        if self.model is None:
            self.model = self._load_model()

        text_features = self.model.encode(query)
        return {"embedding": text_features.tolist(), "meta_data": {}}

    def _load_model(self):
        """Load data from a director of images."""
        # load model
        self.model = SentenceTransformer(self.MODEL_NAME)
