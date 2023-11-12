from typing import Any, Dict, List, Union
import copy

from ..sources.data_source import DataSource
from ..enums import MEDIA_TYPE
from ..types import MediaData
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
            media_type: MEDIA_TYPE,
    ) -> List[str]:
        raise NotImplementedError

    def query(
            self,
            input_query: str,
            input_embeddings: List[float],
            n_results: int,
            media_type: MEDIA_TYPE,
            distance_threshold: float,
            collection_name: str
    ) -> List[MediaData]:
        raise NotImplementedError

    def get_existing_document_ids(
            self, document_ids: List[str], meida_type: MEDIA_TYPE
    ) -> List[str]:
        raise NotImplementedError

    def _construct_metadata(
            self, metadata: List[Dict[str, Any]], source: str, document_id: str, len: int
    ):
        new_metadata = copy.deepcopy(metadata)
        is_metadata_empty = not metadata
        if is_metadata_empty:
            new_metadata = []

        for i in range(len):
            temp_metadata = {
                "source_type": DataSource.LOCAL.name,
                "source_id": source,
                "document_id": document_id,
            }
            if is_metadata_empty:
                new_metadata.append(temp_metadata)
            else:
                new_metadata[i].update(temp_metadata)

        return new_metadata

    def embed_and_store(self, embedding_model, data, media_type, file, source, datasource):
        raise NotImplementedError
