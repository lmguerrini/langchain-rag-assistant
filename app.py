import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.chains import answer_query
from src.config import get_settings


DEFAULT_CHAT_MODEL = "gpt-4.1-mini"


@st.cache_resource
def get_vector_store() -> Chroma:
    settings = get_settings()
    api_key = settings.ensure_openai_api_key()
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model=settings.embedding_model,
    )
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_persist_dir),
    )


@st.cache_resource
def get_chat_model() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        api_key=settings.ensure_openai_api_key(),
        model=DEFAULT_CHAT_MODEL,
        temperature=0,
    )


def render_latest_turn() -> None:
    latest_turn = st.session_state.get("latest_turn")
    if not latest_turn:
        return

    with st.chat_message("user"):
        st.write(latest_turn["query"])

    with st.chat_message("assistant"):
        st.write(latest_turn["answer"])
        if not latest_turn["used_context"]:
            st.caption("This response used the no-context fallback because no relevant chunks were retrieved.")
        if latest_turn["sources"]:
            with st.expander("Sources"):
                for source in latest_turn["sources"]:
                    st.write(f"- {source}")


def main() -> None:
    st.set_page_config(page_title="RAG Assistant", page_icon=":speech_balloon:")
    st.title("RAG Assistant")
    st.write(
        "Ask about LangChain-based RAG application development with Chroma and Streamlit."
    )

    if "latest_turn" not in st.session_state:
        st.session_state["latest_turn"] = None

    render_latest_turn()

    question = st.chat_input("Ask a question about the knowledge base")
    if not question:
        return

    try:
        with st.spinner("Generating answer..."):
            result = answer_query(
                query=question,
                vector_store=get_vector_store(),
                chat_model=get_chat_model(),
            )
    except Exception as exc:
        st.error(str(exc))
        return

    st.session_state["latest_turn"] = {
        "query": question,
        "answer": result.answer,
        "used_context": result.used_context,
        "sources": result.answer_sources,
    }
    st.rerun()


if __name__ == "__main__":
    main()
