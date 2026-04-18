from types import SimpleNamespace

from services.chat_service import get_chat_model
from src.config import SUPPORTED_CHAT_MODELS


class FakeSettings(SimpleNamespace):
    supported_chat_models = SUPPORTED_CHAT_MODELS
    default_chat_model = "gpt-4.1-mini"

    def ensure_openai_api_key(self) -> str:
        return self.openai_api_key

    def ensure_supported_chat_model(self, model_name: str) -> str:
        cleaned_model_name = model_name.strip()
        if cleaned_model_name not in self.supported_chat_models:
            raise ValueError("Unsupported chat model selected.")
        return cleaned_model_name


def test_get_chat_model_restores_retry_and_timeout_hardening() -> None:
    model = get_chat_model(FakeSettings(openai_api_key="key-123"), "gpt-4.1-mini")

    assert model.max_retries == 3
    assert model.request_timeout == 30.0
