from typing import List, Union, Any
from .configs.base import BaseVectorDatabaseConfig
from ..enums import MEDIA_TYPE


class BaseVectorDatabase:
    def __init__(self, config: BaseVectorDatabaseConfig):
        self.config = config
        pass

    def add(self, embeddings: List[List[float]], documents: List[str], ids: List[str], metadata: List[List[Any]]) -> \
            List[str]:
        raise NotImplementedError

    def query(self, input_query: str, input_embeddings: List[float], n_results: int, data_type: MEDIA_TYPE) -> \
    List[str]:
        raise NotImplementedError

    def get_existing_object_identifiers(self, object_identifiers, data_type: MEDIA_TYPE) -> List[str]:
        raise NotImplementedError
