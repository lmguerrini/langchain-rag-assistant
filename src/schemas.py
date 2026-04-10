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
