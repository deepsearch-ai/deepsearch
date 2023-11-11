from typing import Any

from ..enums import MEDIA_TYPE
from ..sources.data_source import DataSource


class BaseEmbeddingModel:
    def __init__(self):
        pass

    def get_media_encoding(
        self, data: Any, dataType: MEDIA_TYPE, datasource: DataSource
    ):
        raise NotImplementedError

    def get_text_encoding(self, query: str):
        raise NotImplementedError
