from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @abstractmethod
    def set_generation_model(self, model_name: str):
        """
        Set the generation model to be used.
        :param model_name: The name of the model to be used.
        """
        pass

    @abstractmethod
    def set_embedding_model(self, model_id: str, model_size: int):
        """
        Set the embedding model to be used.
        :param model_id: The id of the model to be used.
        :param model_size: The size of the embedding model.
        """
        pass

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        chat_history: list,
        max_output_tokens: int,
        temperature: float,
    ) -> str | None:
        """
        Generate text based on the provided prompt.
        :param prompt: The input prompt for text generation.
        :param chat_history: The history of the chat for context.
        :param max_output_tokens: The maximum number of tokens to generate.
        :param temperature: The sampling temperature to use.
        :return: The generated text.
        """
        pass

    @abstractmethod
    def embed_text(self, text: str, document_type: str):
        """
        Embed the provided text.
        :param document_type: The input document type to be embedded.
        :return: The embedded representation of the text.
        """
        pass

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str) -> str:
        """
        Construct a prompt for the LLM.
        :param prompt: The input prompt to be constructed.
        :param role: The role to be included in the prompt.
        :return: The constructed prompt.
        """
        pass
