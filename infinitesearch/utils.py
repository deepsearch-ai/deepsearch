from .sources.utils import SourceUtils
from .llms.base import BaseLLM
from .vector_databases.base import BaseVectorDatabase


class Utils:
    def __init__(self):
        self.source_utils = SourceUtils()

    def add_data(self, source: str, llm_model: BaseLLM, vector_database: BaseVectorDatabase) -> None:
        self.source_utils.add_data(source, llm_model, vector_database)

    def query(self, query: str, llm_model: BaseLLM, vector_database: BaseVectorDatabase) -> None:
        self.source_utils.query(query, llm_model, vector_database)
