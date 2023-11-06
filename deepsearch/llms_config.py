from typing import Optional

from .enums import MEDIA_TYPE
from .embedding_models.base import BaseEmbeddingModel
from .embedding_models.clip import Clip
from .embedding_models.whisper_openai import WhisperOpenAi


class LlmsConfig:
    def __init__(
        self,
        image_llm_model: Optional[BaseEmbeddingModel] = None,
        audio_llm_model: Optional[BaseEmbeddingModel] = None,
    ):
        if not image_llm_model:
            image_llm_model = Clip()
        if not audio_llm_model:
            audio_llm_model = WhisperOpenAi()

        self.llm_models = {
            MEDIA_TYPE.AUDIO: audio_llm_model,
            MEDIA_TYPE.IMAGE: image_llm_model,
        }

    def get_llm_model(self, media_type: MEDIA_TYPE):
        return self.llm_models.get(media_type, None)
