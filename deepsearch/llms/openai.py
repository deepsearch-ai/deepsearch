from .base import BaseLLM
from .configs.openai import OpenAiConfig
from ..enums import MEDIA_TYPE
from ..vector_databases.base import BaseVectorDatabase
from typing import Any, Dict, List
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


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
        prompt = self.DEFAULT_PROMPT_TEMPLATE.substitute(context=context_string, query=input_query)
        return prompt

    def query(self, query: str, vector_database: BaseVectorDatabase,media_types: List[MEDIA_TYPE]):
        response = vector_database.query(input_query=query, input_embeddings=None, n_results=10,
                                              media_types=media_types, distance_threshold=0.5)
        prompt = self.generate_prompt(query, response)
        return self.get_llm_model_answer(prompt)

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
