from typing import List, Union
from .configs.base import BaseVectorDatabaseConfig


class BaseVectorDatabase:
    def __init__(self, config: BaseVectorDatabaseConfig):
        self.config = config
        pass

    def add(self, embeddings: List[List[float]], documents: List[str], ids: List[str]) -> List[str]:
        raise NotImplementedError

    def query(self, input_query: Union[List[float], str], n_results: int) -> List[str]:
        raise NotImplementedError

    def get_existing_object_identifiers(self, object_identifiers) -> List[str]:
        raise NotImplementedError
