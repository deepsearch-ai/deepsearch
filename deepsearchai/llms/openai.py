from typing import Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from deepsearchai.enums import MEDIA_TYPE
from deepsearchai.types import MediaData, QueryResult
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
        contexts: Dict[MEDIA_TYPE, List[MediaData]],
    ) -> QueryResult:
        results = []
        for item in contexts.items():
            media_data = item[1]
            for each_response in media_data:
                results.append(each_response.get("document", ""))
        prompt = self.generate_prompt(query, results)
        llm_response = self.get_llm_model_answer(prompt)
        query_result = {"llm_response": llm_response, "documents": contexts}
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
