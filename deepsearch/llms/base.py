from typing import Any

from ..enums import MEDIA_TYPE


class BaseLLM:
    def __init__(self):
        pass

    def get_media_encoding(self, data: Any, dataType: MEDIA_TYPE):
        raise NotImplementedError

    def get_text_encoding(self, query: str):
        raise NotImplementedError
