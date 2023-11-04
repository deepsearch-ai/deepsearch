from typing import List

import deepsearch.utils as utils

from .embedding_models.base import BaseEmbeddingModel
from .enums import MEDIA_TYPE
from .llms_config import LlmsConfig
from .sources.utils import SourceUtils
from .vector_databases.base import BaseVectorDatabase
from .vector_databases.chromadb import ChromaDB


class App:
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        llms_config: LlmsConfig,
        vector_database: BaseVectorDatabase,
    ):
        self.embedding_model = embedding_model
        self.vector_database = vector_database if vector_database else ChromaDB()
        self.llms_config = llms_config if llms_config else LlmsConfig()
        self.source_utils = SourceUtils()

    def add_data(self, source: str):
        self.source_utils.add_data(source, self.llms_config, self.vector_database)

    def query(self, query: str, data_types: List[MEDIA_TYPE] = [MEDIA_TYPE.IMAGE]):
        return self.source_utils.query(
            query, data_types, self.llms_config, self.vector_database
        )
