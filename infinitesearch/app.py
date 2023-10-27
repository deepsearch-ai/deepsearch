from embedding_models.base import BaseEmbeddingModel
from vector_databases.base import BaseVectorDatabase


class App:
    def __init__(
            self,
            embedding_model: BaseEmbeddingModel,
            database: BaseVectorDatabase,
    ):
        self.embedding_model = embedding_model
        self.database = database

    def add_source(self):
        pass

    def change_interval(self):
        pass

    def add_data(self):
        pass

    def sync(self, source_id):
        pass
