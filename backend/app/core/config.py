from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "FinanceGuru API"
    API_V1_STR: str = "/api/v1"
    
    # Hugging Face Configuration
    HF_MODEL_NAME: str = "microsoft/phi-2"  # Better reasoning and natural language generation
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"  # Higher quality embeddings
    HF_CACHE_DIR: str = "./models"
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "financial_documents"
    
    # RAG Configuration
    RAG_TOP_K: int = 5
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 200
    
    # Legacy Ollama (keep for backward compatibility)
    OLLAMA_API_URL: str = "http://localhost:11434/api/generate"
    
    # model_config allows pydantic to load from .env files
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

settings = Settings()