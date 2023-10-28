from sources import utils as source_utils
from datatype import DataType
from llms.base import BaseLLM


def infer_type(source: str) -> DataType:
    pass


def add_data(source: str, llm_model: BaseLLM) -> DataType:
    source_utils.add_data(source, llm_model)
