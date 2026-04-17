from __future__ import annotations

import streamlit as st

from src.config import Settings


KB_REBUILD_FEEDBACK_KEY = "kb_rebuild_feedback"
CHAT_MODEL_SESSION_KEY = "selected_chat_model"
ANALYTICS_EVAL_REPORT_KEY = "analytics_evaluation_report"
ANALYTICS_EVAL_ERROR_KEY = "analytics_evaluation_error"


def initialize_session_state(settings: Settings) -> None:
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []
    if "request_timestamps" not in st.session_state:
        st.session_state["request_timestamps"] = []
    if CHAT_MODEL_SESSION_KEY not in st.session_state:
        st.session_state[CHAT_MODEL_SESSION_KEY] = settings.default_chat_model
    if ANALYTICS_EVAL_REPORT_KEY not in st.session_state:
        st.session_state[ANALYTICS_EVAL_REPORT_KEY] = None
    if ANALYTICS_EVAL_ERROR_KEY not in st.session_state:
        st.session_state[ANALYTICS_EVAL_ERROR_KEY] = None
