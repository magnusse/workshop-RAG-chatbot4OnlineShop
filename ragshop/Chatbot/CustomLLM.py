
import requests

from typing import List, Optional

# Welches LLM wollen wir für die Chatbot Anfrage nehmen
MODEL_NAME = "Mistral.mistral-medium-latest"
# MODEL_NAME = "IONOS.meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"

class WPSCustomLLM:
    """A custom chat model based on WPS Hamburg CustomLLMs.
    """
    """API-Key for the Interface to WPS LLMs."""
    api_key: str

    def __init__(self, api_key: str):
        self.api_key = api_key

    def call(self,
        messages: List[dict],
        stop: Optional[List[str]] = None
    ) -> str:
        """Run the LLM on the given chat messages.

        Using WPS- Custom LLMs on https://gpt.wps.de/api/chat/completions

        Args:
            messages: A list of OpenAI-style chat messages, each dict with
                "role" (system/user/assistant) and "content" (str).
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.

        Returns:
            The raw response body from the API as a string (JSON).
        """

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        url = 'https://gpt.wps.de/api/chat/completions'
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": MODEL_NAME,
            "messages": messages,
        }
        response = requests.post(url, headers=headers, json=data)
        return response.text


