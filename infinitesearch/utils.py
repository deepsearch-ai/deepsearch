from .sources import utils as source_utils
from .llms.base import BaseLLM
from .vector_databases.base import BaseVectorDatabase


def add_data(source: str, llm_model: BaseLLM, vector_database: BaseVectorDatabase) -> None:
    source_utils.add_data(source, llm_model, vector_database)


def query(query: str, llm_model: BaseLLM, vector_database: BaseVectorDatabase) -> None:
    source_utils.query(query, llm_model, vector_database)
