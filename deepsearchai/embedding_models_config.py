from typing import Optional

from .embedding_models.base import BaseEmbeddingModel
from .embedding_models.clip import Clip
from .embedding_models.whisper_openai import WhisperOpenAi
from .enums import MEDIA_TYPE


class EmbeddingModelsConfig:
    def __init__(
        self,
        image_embedding_model: Optional[BaseEmbeddingModel] = None,
        audio_embedding_model: Optional[BaseEmbeddingModel] = None,
    ):
        if not image_embedding_model:
            image_embedding_model = Clip()
        if not audio_embedding_model:
            audio_embedding_model = WhisperOpenAi()

        self.llm_models = {
            MEDIA_TYPE.AUDIO: audio_embedding_model,
            MEDIA_TYPE.IMAGE: image_embedding_model,
            MEDIA_TYPE.VIDEO: audio_embedding_model,
        }

    def get_embedding_model(self, media_type: MEDIA_TYPE):
        return self.llm_models.get(media_type, None)
