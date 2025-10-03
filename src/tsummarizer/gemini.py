from google.genai import Client, types

from .base import BaseTsumGenerator, Prompt
from src.schema.gemini import GeminiTextPrompt, GeminiYoutubePrompt, GeminiPrompt


class GeminiTsumGenerator(BaseTsumGenerator):
    def __init__(self, api_key: str, model_name: str = "models/gemini-2.5-flash"):
        self.model_name = model_name
        self.api_key = api_key

        if self.api_key is None:
            raise ValueError(
                "API key for Gemini is not set. Please set the 'GEMINI_API_KEY' environment variable."
            )

        self.client = Client(api_key=self.api_key)

    def generate(self, prompts: list[Prompt]) -> str:
        messages = []
        for prompt in prompts:
            messages.append(GeminiYoutubePrompt(video_path=prompt.video_path))
            messages.append(GeminiTextPrompt(message=prompt.text))

        contents = GeminiPrompt(messages=messages)

        response = self.client.models.generate_content(
            model=self.model_name, contents=types.Content(parts=contents.to_parts())
        )

        if response is None or response.text is None:
            raise ValueError("No response from Gemini API.")

        return response.text
