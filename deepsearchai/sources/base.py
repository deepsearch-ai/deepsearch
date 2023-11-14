import copy
import mimetypes
from typing import Any, Dict, List

from ..embedding_models.base import BaseEmbeddingModel
from ..enums import MEDIA_TYPE
from ..vector_databases.base import BaseVectorDatabase
from .data_source import DataSource


class BaseSource:
    def __init__(self):
        pass

    def add_data(
        self,
        source: str,
        llm_model: BaseEmbeddingModel,
        vector_database: BaseVectorDatabase,
    ) -> None:
        raise NotImplementedError