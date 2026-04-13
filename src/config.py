from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


SUPPORTED_CHAT_MODELS = (
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
)


class Settings(BaseSettings):
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    default_chat_model: str = Field(
        default="gpt-4.1-mini",
        alias="DEFAULT_CHAT_MODEL",
    )
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
    official_langchain_docs_mcp_url: str = Field(
        default="https://docs.langchain.com/mcp",
        alias="OFFICIAL_LANGCHAIN_DOCS_MCP_URL",
    )
    official_openai_docs_mcp_url: str = Field(
        default="https://developers.openai.com/mcp",
        alias="OFFICIAL_OPENAI_DOCS_MCP_URL",
    )
    official_docs_timeout_seconds: float = Field(
        default=15.0,
        alias="OFFICIAL_DOCS_TIMEOUT_SECONDS",
    )
    official_docs_fallback_manifest_path: Path = Field(
        default=Path("data/official_docs/source_manifest.json"),
        alias="OFFICIAL_DOCS_FALLBACK_MANIFEST_PATH",
    )
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    max_query_length: int = Field(default=500, alias="MAX_QUERY_LENGTH")
    rate_limit_request_count: int = Field(default=5, alias="RATE_LIMIT_REQUEST_COUNT")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

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

    @property
    def supported_chat_models(self) -> tuple[str, ...]:
        return SUPPORTED_CHAT_MODELS

    def ensure_supported_chat_model(self, model_name: str) -> str:
        cleaned_model_name = model_name.strip()
        if cleaned_model_name not in self.supported_chat_models:
            raise ValueError(
                "Unsupported chat model selected. Choose one of: "
                + ", ".join(self.supported_chat_models)
            )
        return cleaned_model_name

    def model_post_init(self, __context) -> None:
        self.default_chat_model = self.ensure_supported_chat_model(
            self.default_chat_model
        )


def get_settings() -> Settings:
    return Settings()
