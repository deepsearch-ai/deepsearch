from typing import Any
from sentence_transformers import SentenceTransformer
from PIL import Image, UnidentifiedImageError


class Clip:
    MODEL_NAME = "clip-ViT-B-32"

    def __init__(self):
        self.load_model()

    def add_data(self, data: Any, source: str, metadata: Any):
        raise NotImplementedError

    def load_model(self):
        """Load data from a director of images."""
        # load model
        self.model = SentenceTransformer(self.MODEL_NAME)

    def get_media_encoding(self, data: Any):
        """
        Applies the CLIP model to evaluate the vector representation of the supplied image
        """
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
        model = self.load_model()
        text_features = model.encode(query)
        return {"embedding": text_features.tolist(), "meta_data": {}}
