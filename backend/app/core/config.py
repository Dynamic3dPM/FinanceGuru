from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "FinanceGuru API"
    API_V1_STR: str = "/api/v1"
    # Add other configurations here as needed
    # e.g., OLLAMA_API_URL: str = "http://localhost:11434"
    # GOOGLE_APPLICATION_CREDENTIALS: str = "" # Path to your GCP credentials JSON

    # model_config allows pydantic to load from .env files
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

settings = Settings()