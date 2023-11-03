import deepsearch.utils as utils
from typing import List

from .enums import MEDIA_TYPE
from .embedding_models.base import BaseEmbeddingModel
from .llms_config import LlmsConfig
from .vector_databases.base import BaseVectorDatabase
from .sources.utils import SourceUtils

class App:
    def __init__(
            self,
            embedding_model: BaseEmbeddingModel,
            llms_config: LlmsConfig,
            vector_database: BaseVectorDatabase,
    ):
        self.embedding_model = embedding_model
        self.vector_database = vector_database
        self.llms_config = llms_config
        self.source_utils = SourceUtils()

    def add_data(self, source: str):
        self.source_utils.add_data(source, self.llms_config, self.vector_database)

    def query(self, query: str, data_types: List[MEDIA_TYPE] = [MEDIA_TYPE.IMAGE]):
        return self.source_utils.query(query, data_types, self.llms_config, self.vector_database)
