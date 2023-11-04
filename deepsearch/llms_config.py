from typing import Optional

from .enums import MEDIA_TYPE
from .llms.base import BaseLLM
from .llms.clip import Clip
from .llms.whisper import Whisper


class LlmsConfig:
    def __init__(
        self,
        image_llm_model: Optional[BaseLLM] = None,
        audio_llm_model: Optional[BaseLLM] = None,
    ):
        if not image_llm_model:
            image_llm_model = Clip()
        if not audio_llm_model:
            audio_llm_model = Whisper()

        self.llm_models = {
            MEDIA_TYPE.AUDIO: audio_llm_model,
            MEDIA_TYPE.IMAGE: image_llm_model,
        }

    def get_llm_model(self, media_type: MEDIA_TYPE):
        return self.llm_models.get(media_type, None)
