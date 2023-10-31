from .vector_databases.base import BaseVectorDatabase
from .llms.base import BaseLLM
from .embedding_models.base import BaseEmbeddingModel
from .utils import Utils

import infinitesearch.utils as utils


class App:
    def __init__(
            self,
            embedding_model: BaseEmbeddingModel,
            llm_model: BaseLLM,
            vector_database: BaseVectorDatabase,
    ):
        self.embedding_model = embedding_model
        self.vector_database = vector_database
        self.llm_model = llm_model
        self.utils = Utils()

    def add_source(self):
        pass

    def change_sync_frequency(self):
        pass

    def add_data(self, source: str):
        self.utils.add_data(source, self.llm_model, self.vector_database)
        pass

    def query(self, query: str):
        self.utils.query(query, self.llm_model, self.vector_database)
        pass

    def sync(self, source_id):
        pass
