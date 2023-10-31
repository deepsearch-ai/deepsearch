from ..llms.base import BaseLLM
from ..vector_databases.base import BaseVectorDatabase


class BaseSource:
    def __init__(self):
        pass

    def add_data(self, source: str, llm_model: BaseLLM, vector_database: BaseVectorDatabase) -> None:
        raise NotImplementedError
