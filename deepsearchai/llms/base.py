from string import Template
from typing import Dict, List

from deepsearchai.enums import MEDIA_TYPE
from deepsearchai.types import MediaData, QueryResult

DEFAULT_PROMPT = """
  Use the following pieces of context to answer the query at the end.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.

  $context

  Query: $query

  Helpful Answer:
"""  # noqa:E501


class BaseLLM:
    DEFAULT_PROMPT_TEMPLATE = Template(DEFAULT_PROMPT)

    def __init__(self):
        pass

    def query(
        self,
        query: str,
        contexts: Dict[MEDIA_TYPE, List[MediaData]],
    ) -> QueryResult:
        raise NotImplementedError
