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
