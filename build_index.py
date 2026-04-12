from src.config import get_settings
from src.kb_status import write_kb_manifest
from src.knowledge_base import KnowledgeBaseError, build_index


def main() -> None:
    settings = get_settings()
    vector_store = build_index(settings=settings)
    indexed_count = len(vector_store.get()["ids"])
    manifest_path = write_kb_manifest(
        settings=settings,
        indexed_chunk_count=indexed_count,
    )
    print(
        "Indexed "
        f"{indexed_count} chunks into '{settings.chroma_collection_name}' "
        f"at {settings.chroma_persist_dir}"
    )
    print(f"Wrote KB manifest to {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KnowledgeBaseError as exc:
        raise SystemExit(f"Knowledge base build failed: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
