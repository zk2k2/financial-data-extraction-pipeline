from enum import Enum


class LLMEnums(Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    AZURE = "azure"
    LOCAL = "local"


class OpenAIEnums(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
