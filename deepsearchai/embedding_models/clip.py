import uuid
from typing import Any

from sentence_transformers import SentenceTransformer

from ..enums import MEDIA_TYPE
from ..sources.data_source import DataSource
from .base import BaseEmbeddingModel


class Clip(BaseEmbeddingModel):
    MODEL_NAME = "clip-ViT-B-32"
    SUPPORTED_MEDIA_TYPES = [MEDIA_TYPE.IMAGE]

    def __init__(self):
        self._load_model()

    def get_media_encoding(
        self, data: Any, data_type: MEDIA_TYPE, datasource: DataSource
    ):
        """
        Applies the CLIP model to evaluate the vector representation of the supplied image
        """
        if data_type not in self.SUPPORTED_MEDIA_TYPES:
            raise ValueError(
                "Unsupported dataType. Clip model supports only {}".format(
                    self.SUPPORTED_MEDIA_TYPES
                )
            )
        image_features = self.model.encode(data)
        return {"embedding": [image_features.tolist()], "ids": [str(uuid.uuid4())]}

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
