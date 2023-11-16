from typing import Optional

from deepsearchai.embedding_models.base import BaseEmbeddingModel
from deepsearchai.embedding_models.blip_image_captioning import BlipImageCaptioning
from deepsearchai.embedding_models.clip import Clip
from deepsearchai.embedding_models.whisper_model import Whisper
from deepsearchai.enums import MEDIA_TYPE


class EmbeddingModelsConfig:
    def __init__(
        self,
        image_embedding_model: Optional[BaseEmbeddingModel] = None,
        audio_embedding_model: Optional[BaseEmbeddingModel] = None,
        video_embedding_model: Optional[BaseEmbeddingModel] = None,
        image_captioning_model: Optional[BaseEmbeddingModel] = BlipImageCaptioning(),
    ):
        if not image_embedding_model:
            image_embedding_model = Clip()
        if not audio_embedding_model:
            audio_embedding_model = Whisper()
        if not video_embedding_model:
            video_embedding_model = Whisper()
        image_embedding_models = [image_embedding_model]
        audio_embedding_models = [audio_embedding_model]
        video_embedding_models = [video_embedding_model]
        if image_captioning_model:
            image_embedding_models.append(image_captioning_model)

        self.llm_models = {
            MEDIA_TYPE.AUDIO: audio_embedding_models,
            MEDIA_TYPE.IMAGE: image_embedding_models,
            MEDIA_TYPE.VIDEO: video_embedding_models,
        }

    def get_embedding_model(self, media_type: MEDIA_TYPE):
        return self.llm_models.get(media_type, [])
