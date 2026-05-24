"""Client for the personal LLM Gateway native chat endpoint."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from config.settings import Config

logger = logging.getLogger(__name__)


class LLMGatewayAdapter:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def _headers(self) -> dict[str, str]:
        if not self.config.LLM_GATEWAY_API_KEY:
            raise RuntimeError("LLM_GATEWAY_API_KEY is not configured.")
        return {
            "Content-Type": "application/json",
            "X-API-Key": self.config.LLM_GATEWAY_API_KEY,
        }

    @staticmethod
    def _encode_image(image_path: str | Path) -> tuple[str, str]:
        path = Path(image_path)
        media_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        with path.open("rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
        return encoded, media_type

    @staticmethod
    def _prompts_from_messages(messages: list[dict[str, str]]) -> tuple[str | None, str]:
        system_parts: list[str] = []
        user_parts: list[str] = []

        for message in messages:
            role = message.get("role", "user")
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                user_parts.append(content)
            else:
                user_parts.append(f"{role}: {content}")

        if not user_parts:
            raise ValueError("LLM Gateway chat requires at least one user message.")
        return "\n\n".join(system_parts) or None, "\n\n".join(user_parts)

    @staticmethod
    def extract_content(response: dict[str, Any]) -> str:
        content = response.get("content")
        if isinstance(content, str):
            return content
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM Gateway response did not contain message content.") from exc

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        image_path: str | Path | None = None,
        image_base64: str | None = None,
        image_media_type: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        system_prompt, user_prompt = self._prompts_from_messages(messages)

        config_payload: dict[str, Any] = {"model": model}
        for key in ("temperature", "max_output_tokens", "top_p"):
            value = kwargs.pop(key, None)
            if value is not None:
                config_payload[key] = value
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None and "max_output_tokens" not in config_payload:
            config_payload["max_output_tokens"] = max_tokens
        if kwargs:
            config_payload["extra"] = kwargs

        payload: dict[str, Any] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "config": config_payload,
        }

        if image_path is not None:
            image_base64, image_media_type = self._encode_image(image_path)
        if image_base64:
            payload["image_base64"] = image_base64
            payload["image_media_type"] = image_media_type or "image/jpeg"

        url = f"{self.config.LLM_GATEWAY_URL.rstrip('/')}{self.config.LLM_GATEWAY_CHAT_PATH}"
        logger.debug("POST %s model=%s image=%s", url, model, bool(image_base64))
        timeout = httpx.Timeout(self.config.LLM_GATEWAY_TIMEOUT_SECONDS, connect=10.0)
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload, headers=self._headers())
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:500]
            raise RuntimeError(
                f"LLM Gateway request failed with {response.status_code}: {detail}"
            ) from exc
        return response.json()

