import mimetypes
from typing import Any, Dict

from ..enums import MEDIA_TYPE
from ..llms.base import BaseLLM
from ..vector_databases.base import BaseVectorDatabase
from .data_source import DataSource


class BaseSource:
    def __init__(self):
        pass

    def add_data(
        self, source: str, llm_model: BaseLLM, vector_database: BaseVectorDatabase
    ) -> None:
        raise NotImplementedError

    def _construct_metadata(
        self, metadata: Dict[str, Any], source: str, document_id: str, len: int
    ):
        is_metadata_empty = not metadata
        if is_metadata_empty:
            metadata = []

        for i in range(len):
            temp_metadata = {
                "source_type": DataSource.LOCAL.name,
                "source_id": source,
                "document_id": document_id,
            }
            if is_metadata_empty:
                metadata.append(temp_metadata)
            else:
                metadata[i].update(temp_metadata)

        return metadata
