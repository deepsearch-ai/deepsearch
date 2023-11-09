from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from typing_extensions import TypedDict

from ..enums import MEDIA_TYPE
from ..types import MediaData, QueryResult
from ..vector_databases.base import BaseVectorDatabase
from .base import BaseLLM
from .configs.openai import OpenAiConfig


class OpenAi(BaseLLM):
    def __init__(self, config: OpenAiConfig = OpenAiConfig()):
        self.config = config
        super().__init__()

    def generate_prompt(self, input_query: str, contexts: List[str]) -> str:
        """
        Generates a prompt based on the given query and context, ready to be
        passed to an LLM

        :param input_query: The query to use.
        :type input_query: str
        :param contexts: List of similar documents to the query used as context.
        :type contexts: List[str]
        :return: The prompt
        :rtype: str
        """
        context_string = (" | ").join(contexts)
        # basic use case, no history.
        prompt = self.DEFAULT_PROMPT_TEMPLATE.substitute(
            context=context_string, query=input_query
        )
        return prompt

    def query(
        self,
        query: str,
        vector_database: BaseVectorDatabase,
        media_types: List[MEDIA_TYPE],
    ) -> QueryResult:
        media_data = {}
        results = []
        for media_type in media_types:
            response = vector_database.query(
                input_query=query,
                input_embeddings=None,
                n_results=10,
                media_type=media_type,
                distance_threshold=12,
            )
            media_data[media_type] = response
            for each_response in response:
                results.extend(each_response.get("document", ""))
        prompt = self.generate_prompt(query, results)
        llm_response = self.get_llm_model_answer(prompt)

        # MediaData = TypedDict("MediaData", {'document':str, 'metadata':dict})
        # QueryResult = TypedDict("QueryResult", {'llm_response':str, 'documents':List[MediaData]})
        query_result = {"llm_response": llm_response, "documents": media_data}
        return query_result

    def get_llm_model_answer(self, prompt) -> str:
        response = self._get_answer(prompt, self.config)
        return response

    def _get_answer(self, prompt: str, config: OpenAiConfig) -> str:
        messages = []
        messages.append(HumanMessage(content=prompt))
        kwargs = {
            "model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "model_kwargs": {},
        }
        chat = ChatOpenAI(**kwargs)
        return chat(messages).content
