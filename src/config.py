from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="EMBEDDING_MODEL",
    )
    chroma_persist_dir: Path = Field(
        default=Path("data/chroma_db"),
        alias="CHROMA_PERSIST_DIR",
    )
    chroma_collection_name: str = Field(
        default="langchain_rag_knowledge_base",
        alias="CHROMA_COLLECTION_NAME",
    )
    raw_data_dir: Path = Field(default=Path("data/raw"), alias="RAW_DATA_DIR")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    def ensure_openai_api_key(self) -> str:
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required to build the Chroma index with "
                "OpenAI embeddings. Add it to your environment or .env file."
            )
        return self.openai_api_key


def get_settings() -> Settings:
    return Settings()
