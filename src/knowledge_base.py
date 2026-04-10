from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.config import Settings
from src.schemas import ChunkMetadata, DocumentMetadata


FRONTMATTER_BOUNDARY = "---"
DOC_ID_PATTERN = re.compile(r"[^a-z0-9]+")


class KnowledgeBaseError(ValueError):
    pass


@dataclass
class LoadedMarkdownDocument:
    metadata: DocumentMetadata
    body: str


def load_markdown_documents(raw_dir: Path) -> list[LoadedMarkdownDocument]:
    if not raw_dir.exists():
        raise KnowledgeBaseError(f"Knowledge-base directory does not exist: {raw_dir}")

    files = sorted(path for path in raw_dir.iterdir() if path.is_file())
    unsupported_files = [path.name for path in files if path.suffix.lower() != ".md"]
    if unsupported_files:
        raise KnowledgeBaseError(
            "Unsupported files found in data/raw/: "
            + ", ".join(unsupported_files)
            + ". Only .md files are supported."
        )

    markdown_files = [path for path in files if path.suffix.lower() == ".md"]
    if not markdown_files:
        raise KnowledgeBaseError(
            f"No markdown files found in knowledge-base directory: {raw_dir}"
        )

    return [_load_single_markdown_document(path) for path in markdown_files]


def split_documents(
    documents: list[LoadedMarkdownDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Document] = []

    for loaded_document in documents:
        split_texts = splitter.split_text(loaded_document.body)
        for chunk_index, chunk_text in enumerate(split_texts):
            chunk_metadata = ChunkMetadata(
                **loaded_document.metadata.model_dump(),
                chunk_index=chunk_index,
            )
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata.model_dump(exclude_none=True),
                )
            )

    return chunks


def build_index(
    settings: Settings,
    embeddings: Embeddings | None = None,
    reset_collection: bool = True,
) -> Chroma:
    documents = load_markdown_documents(settings.raw_data_dir)
    chunks = split_documents(
        documents=documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise KnowledgeBaseError("No chunks were created from the markdown corpus.")

    embedding_function = embeddings or _build_openai_embeddings(settings)
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

    if reset_collection:
        existing_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embedding_function,
            persist_directory=str(settings.chroma_persist_dir),
        )
        try:
            existing_store.delete_collection()
        except Exception:
            pass

    vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embedding_function,
        persist_directory=str(settings.chroma_persist_dir),
    )
    ids = [_build_chunk_id(chunk.metadata) for chunk in chunks]
    vector_store.add_documents(documents=chunks, ids=ids)
    return vector_store


def _build_openai_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=settings.ensure_openai_api_key(),
        model=settings.embedding_model,
    )


def _load_single_markdown_document(path: Path) -> LoadedMarkdownDocument:
    raw_text = path.read_text(encoding="utf-8").strip()
    frontmatter, body = _split_frontmatter(raw_text, path)
    if "title" not in frontmatter or not frontmatter["title"].strip():
        raise KnowledgeBaseError(
            f"Markdown document is missing required title metadata: {path}"
        )

    document_metadata = DocumentMetadata(
        doc_id=_build_doc_id(path),
        source_path=path.as_posix(),
        title=frontmatter["title"],
        topic=frontmatter["topic"],
        library=frontmatter["library"],
        doc_type=frontmatter["doc_type"],
        difficulty=frontmatter["difficulty"],
        error_family=frontmatter.get("error_family"),
    )
    if not body.strip():
        raise KnowledgeBaseError(f"Markdown document has no body content: {path}")

    return LoadedMarkdownDocument(metadata=document_metadata, body=body.strip())


def _split_frontmatter(raw_text: str, path: Path) -> tuple[dict[str, str], str]:
    if not raw_text.startswith(FRONTMATTER_BOUNDARY):
        raise KnowledgeBaseError(
            f"Markdown document must start with YAML-style frontmatter: {path}"
        )

    parts = raw_text.split(FRONTMATTER_BOUNDARY, 2)
    if len(parts) < 3:
        raise KnowledgeBaseError(f"Markdown frontmatter is incomplete: {path}")

    metadata_block = parts[1].strip()
    body = parts[2].strip()
    metadata: dict[str, str] = {}

    for line in metadata_block.splitlines():
        if ":" not in line:
            raise KnowledgeBaseError(
                f"Invalid frontmatter line in {path}: {line}"
            )
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    return metadata, body


def _build_doc_id(path: Path) -> str:
    normalized = DOC_ID_PATTERN.sub("-", path.stem.lower()).strip("-")
    return normalized or "document"


def _build_chunk_id(metadata: dict[str, object]) -> str:
    return f"{metadata['doc_id']}::chunk::{metadata['chunk_index']}"
