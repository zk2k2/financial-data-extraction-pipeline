from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str

    GENERATION_BACKEND: str
    OPENAI_API_KEY: str
    OPENAI_API_URL: str

    INPUT_DEFAULT_MAX_CHARACTERS: int
    GENERATION_DEFAULT_MAX_TOKENS: int
    DEFAULT_TEMPERATURE: float
    GENERATION_MODEL_ID: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def get_settings() -> Settings:
    return Settings()
