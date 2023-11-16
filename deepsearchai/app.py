from typing import Dict, List

from deepsearchai.embedding_models_config import EmbeddingModelsConfig
from deepsearchai.enums import MEDIA_TYPE
from deepsearchai.llms.base import BaseLLM
from deepsearchai.llms.openai import OpenAi
from deepsearchai.sources.utils import SourceUtils
from deepsearchai.types import MediaData, QueryResult
from deepsearchai.vector_databases.base import BaseVectorDatabase
from deepsearchai.vector_databases.chromadb import ChromaDB


class App:
    def __init__(
        self,
        embedding_models_config: EmbeddingModelsConfig,
        vector_database: BaseVectorDatabase,
        llm: BaseLLM,
    ):
        self.embedding_models_config = (
            embedding_models_config
            if embedding_models_config
            else EmbeddingModelsConfig()
        )
        self.vector_database = (
            vector_database
            if vector_database
            else ChromaDB(embedding_models_config=self.embedding_models_config)
        )

        self.llm = llm if llm else OpenAi(self.vector_database)
        self.source_utils = SourceUtils()

    def add_data(self, source: str):
        self.source_utils.add_data(
            source, self.embedding_models_config, self.vector_database
        )

    def query(
        self, query: str, media_types: List[MEDIA_TYPE] = [MEDIA_TYPE.IMAGE]
    ) -> QueryResult:
        data = self.get_data(query, media_types)
        response = self.llm.query(query, data)
        return response

    def get_data(
        self, query: str, media_types: List[MEDIA_TYPE] = [MEDIA_TYPE.IMAGE]
    ) -> Dict[MEDIA_TYPE, List[MediaData]]:
        return self.source_utils.get_data(
            query, media_types, self.embedding_models_config, self.vector_database
        )
