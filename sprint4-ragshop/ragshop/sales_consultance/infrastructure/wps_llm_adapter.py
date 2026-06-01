import json
from typing import List

import requests

from ragshop.sales_consultance.domain.ports.llm_port import LLMPort


MODEL_NAME = "Mistral.mistral-medium-latest"
_API_URL = "https://gpt.wps.de/api/chat/completions"


class WpsLLMAdapter(LLMPort):
    def __init__(self, api_key: str, model_name: str = MODEL_NAME):
        self._api_key = api_key
        self._model_name = model_name

    def generate(self, messages: List[dict]) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self._model_name, "messages": messages}

        # Domain Story Step 6: generate prompt AND send to LLM
        raw = requests.post(_API_URL, headers=headers, json=payload).text

        # Domain Story Step 7: LLM responds with Answer
        response = json.loads(raw)
        return response["choices"][0]["message"]["content"]
