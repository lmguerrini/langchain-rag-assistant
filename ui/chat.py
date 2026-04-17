from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from services.chat_service import run_streaming_grounded_query
from src.schemas import AnswerResult


def build_chat_input_visibility_script() -> str:
    return """
    <script>
    const parentWindow = window.parent;
    const parentDocument = parentWindow.document;

    function syncChatInputVisibility() {
      const chatInput = parentDocument.querySelector('div[data-testid="stChatInput"]');
      const tabButtons = Array.from(parentDocument.querySelectorAll('button[role="tab"]'));
      const chatTab = tabButtons.find((button) => button.textContent.trim() === 'Chat');
      const analyticsTab = tabButtons.find((button) => button.textContent.trim() === 'Analytics');
      if (!chatInput || !chatTab || !analyticsTab) {
        return;
      }
      const shouldShow = chatTab.getAttribute('aria-selected') === 'true';
      chatInput.style.display = shouldShow ? '' : 'none';
    }

    if (parentWindow.__ragAssistantChatInputObserver) {
      parentWindow.__ragAssistantChatInputObserver.disconnect();
    }

    const observer = new MutationObserver(syncChatInputVisibility);
    observer.observe(parentDocument.body, {
      attributes: true,
      childList: true,
      subtree: true,
    });
    parentWindow.__ragAssistantChatInputObserver = observer;
    syncChatInputVisibility();
    </script>
    """


def render_chat_input_visibility_controller() -> None:
    components.html(build_chat_input_visibility_script(), height=0, width=0)


def render_streaming_grounded_answer(
    *,
    query: str,
    vector_store,
    chat_model,
) -> AnswerResult:
    streamed_tokens: list[str] = []
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()

        def render_token(token: str) -> None:
            streamed_tokens.append(token)
            answer_placeholder.write("".join(streamed_tokens))

        result = run_streaming_grounded_query(
            query=query,
            vector_store=vector_store,
            chat_model=chat_model,
            on_token=render_token,
        )
        if not streamed_tokens:
            answer_placeholder.write(result.answer)
        return result
