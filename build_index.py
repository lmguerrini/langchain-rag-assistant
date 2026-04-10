from src.config import get_settings
from src.knowledge_base import KnowledgeBaseError, build_index


def main() -> None:
    settings = get_settings()
    vector_store = build_index(settings=settings)
    indexed_count = len(vector_store.get()["ids"])
    print(
        "Indexed "
        f"{indexed_count} chunks into '{settings.chroma_collection_name}' "
        f"at {settings.chroma_persist_dir}"
    )


if __name__ == "__main__":
    try:
        main()
    except KnowledgeBaseError as exc:
        raise SystemExit(f"Knowledge base build failed: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
