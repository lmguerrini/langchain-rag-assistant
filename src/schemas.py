from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


Topic = Literal["langchain", "rag", "chroma", "streamlit", "tool_calling", "prompting"]
Library = Literal["langchain", "chroma", "streamlit", "openai", "general"]
DocType = Literal["concept", "how_to", "example", "troubleshooting"]
Difficulty = Literal["intro", "intermediate", "advanced"]
ErrorFamily = Literal["imports", "api", "retrieval", "ui", "persistence"]


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    doc_id: str
    source_path: str
    title: str
    topic: Topic
    library: Library
    doc_type: DocType
    difficulty: Difficulty
    error_family: ErrorFamily | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        if not value:
            raise ValueError("Document metadata must include a non-empty title.")
        return value


class ChunkMetadata(DocumentMetadata):
    chunk_index: int


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 3

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Retrieval query must not be empty.")
        return cleaned

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value < 1 or value > 10:
            raise ValueError("top_k must be between 1 and 10.")
        return value


class RetrievalFilters(BaseModel):
    topic: Topic | None = None
    library: Library | None = None
    doc_type: DocType | None = None
    error_family: ErrorFamily | None = None

    def as_chroma_filter(self) -> dict[str, object]:
        filters = {
            key: value
            for key, value in self.model_dump().items()
            if value is not None
        }
        if not filters:
            return {}
        if len(filters) == 1:
            key, value = next(iter(filters.items()))
            return {key: value}
        return {
            "$and": [{key: value} for key, value in filters.items()]
        }


class RetrievedChunk(BaseModel):
    content: str
    metadata: ChunkMetadata


class RetrievalResult(BaseModel):
    rewritten_query: str
    applied_filters: RetrievalFilters
    used_fallback: bool
    chunks: list[RetrievedChunk]
    sources: list[str]
