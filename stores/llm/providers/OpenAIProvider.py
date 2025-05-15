from stores.llm.LLMEnums import OpenAIEnums
from ..LLMInterface import LLMInterface
from openai import OpenAI, chat


from typing import Optional
import logging


class OpenAIProvider(LLMInterface):
    def __init__(
        self,
        api_key: str,
        api_url: Optional[str] = None,  # support for custom API URL
        default_input_max_characters: int = 1000,  # safe limit
        default_output_max_tokens: int = 1000,  # default output token limit
        default_temperature: float = 0.1,  # not so creative
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url

        self.default_input_max_characters = default_input_max_characters
        self.default_output_max_tokens = default_output_max_tokens
        self.default_temperature = default_temperature
        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None  # size of the embedding vector

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
        )

        self.logger = logging.getLogger(__name__)  # good monitoring

    def set_generation_model(self, generation_model_id: str):
        """
        Set the generation model to be used.
        :param generation_model_id: The id of the gen_model to be used.
        """
        self.generation_model_id = generation_model_id

    def set_embedding_model(self, model_id: str, embedding_size: int | None):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return (
            text[: self.default_output_max_tokens].strip()
            if len(text) > self.default_output_max_tokens
            else text
        )

    def generate_text(
        self,
        prompt: str,
        chat_history: list = [],
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str | None:
        # handling edge cases
        if not self.client:
            self.logger.error("OpenAI client is not initialized.")
            return None
        if not self.generation_model_id:
            self.logger.error("Generation model is not set.")
            return None

        max_output_tokens = max_output_tokens or self.default_output_max_tokens
        temperature = temperature or self.default_temperature

        chat_history.append(
            self.construct_prompt(
                prompt=prompt,
                role=OpenAIEnums.USER.value,
            )
        )

        # Generate the response
        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=chat_history,
                max_tokens=max_output_tokens,
                temperature=temperature,
            )
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return None
        # Check if the response is valid
        if (
            not response
            or not response.choices
            or len(response.choices) == 0
            or not response.choices[0].message
            or not response.choices[0].message.content
        ):
            self.logger.error("Failed to get response from OpenAI.")
            return None

        # Extract the generated text from the response
        return response.choices[0].message.content

    def embed_text(self, text: str, document_type: str):
        # handling edge cases
        if not self.client:
            self.logger.error("OpenAI client is not initialized.")
            return None
        if not self.embedding_model_id:
            self.logger.error("Embedding model is not set.")
            return None

        response = self.client.embeddings.create(
            model=self.embedding_model_id,
            input=text,
        )

        if (
            not response
            or not response.data
            or len(response.data) == 0
            or not response.data[0].embedding
        ):
            self.logger.error("Failed to get embedding from OpenAI.")
            return None

        return response.data[0].embedding

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": self.process_text(prompt)}
