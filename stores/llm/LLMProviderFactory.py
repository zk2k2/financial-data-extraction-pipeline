from helpers.config import Settings
from stores.llm import LLMEnums
from stores.llm.providers import OpenAIProvider


class LLMProviderFactory:
    def __init__(self, config: Settings):
        self.config = config

    def create(self, provider: str):
        if provider == LLMEnums.LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key=self.config.OPENAI_API_KEY,
                api_url=self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_output_max_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_temperature=self.config.DEFAULT_TEMPERATURE,
            )

        else:
            return None
