from typing import Any, List, Union

from ..enums import MEDIA_TYPE
from .configs.base import BaseVectorDatabaseConfig


class BaseVectorDatabase:
    def __init__(self, config: BaseVectorDatabaseConfig):
        self.config = config
        pass

    def add(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        ids: List[str],
        metadata: List[Any],
        data_type: MEDIA_TYPE,
    ) -> List[str]:
        raise NotImplementedError

    def query(
        self,
        input_query: str,
        input_embeddings: List[float],
        n_results: int,
        media_type: List[MEDIA_TYPE],
        distance_threshold: float,
    ) -> List[str]:
        raise NotImplementedError

    def get_existing_document_ids(
        self, document_ids: List[str], data_type: MEDIA_TYPE
    ) -> List[str]:
        raise NotImplementedError
