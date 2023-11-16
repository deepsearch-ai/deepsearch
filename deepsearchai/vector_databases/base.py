import copy
from typing import Any, Dict, List, Union

from deepsearchai.embedding_models.base import BaseEmbeddingModel
from deepsearchai.enums import MEDIA_TYPE
from deepsearchai.sources.data_source import DataSource
from deepsearchai.types import MediaData
from .configs.base import BaseVectorDatabaseConfig


class BaseVectorDatabase:
    def __init__(self, config: BaseVectorDatabaseConfig):
        self.config = config
        pass

    def add(
        self,
        data: Any,
        datasource: DataSource,
        file: str,
        source: str,
        media_type: MEDIA_TYPE,
        embedding_model: BaseEmbeddingModel,
    ):
        raise NotImplementedError

    def query(
        self,
        query: str,
        n_results: int,
        media_type: MEDIA_TYPE,
        distance_threshold: float,
        embedding_model: BaseEmbeddingModel,
    ) -> List[MediaData]:
        raise NotImplementedError

    def get_existing_document_ids(
        self, document_ids: List[str], collection_name: str
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
