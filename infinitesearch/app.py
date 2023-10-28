import utils
from embedding_models.base import BaseEmbeddingModel
from vector_databases.base import BaseVectorDatabase
from llms.base import BaseLLM


class App:
    def __init__(
            self,
            embedding_model: BaseEmbeddingModel,
            llm_model: BaseLLM,
            database: BaseVectorDatabase,
    ):
        self.embedding_model = embedding_model
        self.database = database
        self.llm_model = llm_model

    def add_source(self):
        pass

    def change_sync_frequency(self):
        pass

    def add_data(self, datasource: str):
        datasource = utils.add_data(datasource, self.llm_model)
        pass

    def sync(self, source_id):
        pass
