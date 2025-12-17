#!/usr/bin/env python3
"""
model_client.py

A process-agnostic client for interacting with OpenAI-compatible LLM APIs.
Supports both standard and reasoning-focused models, handling the quirks of
reasoning outputs (e.g., <think> blocks, markdown fences, list-based content
parts) while remaining backward compatible with existing non-reasoning usage.

Usage:
    from model_client import ModelClient

    client = ModelClient()
    response = client.complete_json(
        system_prompt="You are a helpful assistant.",
        user_content="What is 2+2?",
    )
    print(response)

Configuration:
    The client can be configured via:
    1. Constructor parameters
    2. Environment variables (NOVITA_API_KEY, NOVITA_MODEL, NOVITA_API_BASE_URL, NOVITA_KEY_FILE)
    3. Optional JSON/YAML configuration file
    4. Module defaults

Environment Variables:
    NOVITA_API_KEY: API key string (overrides file-based key)
    NOVITA_MODEL: Model identifier
    NOVITA_API_BASE_URL: API base URL
    NOVITA_KEY_FILE: Path to API key file
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_MODEL = "qwen/qwen3-235b-a22b-instruct-2507"
DEFAULT_API_BASE_URL = "https://api.novita.ai/v3/openai"
DEFAULT_KEY_FILE = "keys/novita/key"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 2.0
DEFAULT_PARSE_RETRIES = 2
DEFAULT_PARSE_RETRY_DELAY = 1.0
DEFAULT_REQUEST_TIMEOUT: float | None = None
DEFAULT_RESPONSE_FORMAT = {"type": "json_object"}
RATE_LIMIT_BACKOFF_SECONDS = [2, 4, 8, 16, 32]
RATE_LIMIT_MAX_ATTEMPTS = len(RATE_LIMIT_BACKOFF_SECONDS) + 1

# Environment variable names
ENV_API_KEY = "NOVITA_API_KEY"
ENV_MODEL = "NOVITA_MODEL"
ENV_API_BASE_URL = "NOVITA_API_BASE_URL"
ENV_KEY_FILE = "NOVITA_KEY_FILE"


class ModelType(Enum):
    """Enumeration of supported model categories."""

    STANDARD = "standard"
    REASONING = "reasoning"


@dataclass
class ModelSpec:
    """Specification for a model's behavior and recommended parameters."""

    identifier: str
    model_type: ModelType
    default_temperature: float
    default_top_p: float | None = None
    default_max_tokens: int | None = None
    supports_reasoning_content: bool = False
    uses_think_tags: bool = False


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "qwen/qwen3-235b-a22b-instruct-2507": ModelSpec(
        identifier="qwen/qwen3-235b-a22b-instruct-2507",
        model_type=ModelType.STANDARD,
        default_temperature=DEFAULT_TEMPERATURE,
    ),
    "openai/gpt-oss-120b": ModelSpec(
        identifier="openai/gpt-oss-120b",
        model_type=ModelType.REASONING,
        default_temperature=0.0,
        default_top_p=0.5,
        default_max_tokens=32768,
        supports_reasoning_content=True,
    ),
    "deepseek/deepseek-v3.2-exp": ModelSpec(
        identifier="deepseek/deepseek-v3.2-exp",
        model_type=ModelType.REASONING,
        default_temperature=0.0,
        default_max_tokens=60000,
        uses_think_tags=True,
    ),
    "moonshotai/kimi-k2-thinking": ModelSpec(
        identifier="moonshotai/kimi-k2-thinking",
        model_type=ModelType.REASONING,
        default_temperature=0.0,
        default_top_p=0.75,
        default_max_tokens=32000,
        uses_think_tags=True,
    ),
}


class ModelClientError(Exception):
    """Base exception for model client errors."""


class APIKeyError(ModelClientError):
    """Raised when there's an issue with the API key."""


class ModelCallError(ModelClientError):
    """Raised when the model call fails after retries."""


class ResponseParseError(ModelClientError):
    """Raised when the model response cannot be parsed."""


class ReasoningExtractionError(ModelClientError):
    """Raised when reasoning content cannot be extracted."""


class ConfigError(ModelClientError):
    """Raised when configuration loading fails."""


@dataclass
class ModelClientConfig:
    """Configuration for ModelClient.

    Attributes:
        model: Model identifier to use for API calls.
        api_base_url: Base URL for the OpenAI-compatible API.
        api_key: Direct API key string.
        api_key_file: Path to the API key file.
        temperature: Sampling temperature for model responses.
        top_p: Optional nucleus sampling value (used primarily for reasoning models).
        max_tokens: Optional max_tokens override to send with requests.
        use_reasoning_model: Whether to treat the configured model as a reasoning model.
        capture_reasoning: Whether to capture and return reasoning content when available.
        max_retries: Maximum number of retry attempts for failed API calls.
        retry_base_delay: Base delay (seconds) used for exponential backoff.
        parse_retries: JSON parse retry attempts.
        parse_retry_delay: Delay between parse retry attempts.
        request_timeout: Optional request timeout (seconds).
        response_format: Default response format passed to the API.
    """

    model: str = DEFAULT_MODEL
    api_base_url: str = DEFAULT_API_BASE_URL
    api_key: str | None = None
    api_key_file: str | Path = DEFAULT_KEY_FILE
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float | None = None
    max_tokens: int | None = None
    use_reasoning_model: bool | None = None
    capture_reasoning: bool | None = None
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY
    parse_retries: int = DEFAULT_PARSE_RETRIES
    parse_retry_delay: float = DEFAULT_PARSE_RETRY_DELAY
    request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT
    response_format: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_RESPONSE_FORMAT))


@dataclass
class ModelResponse:
    """Response from a model completion request."""

    content: str
    parsed: Dict[str, Any] | None = None
    reasoning_content: str | None = None


def _load_config_file(config_path: str | Path | None) -> Dict[str, Any]:
    """Load configuration from a JSON or YAML file.

    Args:
        config_path: Path to JSON or YAML configuration file.

    Returns:
        Parsed configuration dictionary (empty if no path provided).

    Raises:
        ConfigError: If the file cannot be read or parsed, or is invalid.
    """
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found at '{path}'")

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Failed to read configuration file '{path}': {exc}") from exc

    try:
        if path.suffix.lower() in {".yml", ".yaml"}:
            try:
                import yaml
            except ImportError as exc:  # noqa: BLE001
                raise ConfigError("PyYAML is required for YAML configuration files.") from exc
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Failed to parse configuration file '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file '{path}' must contain a JSON/YAML object.")

    return data


def _resolve_value(
    override: Any,
    env_value: Any,
    config_value: Any,
    default_value: Any,
) -> Any:
    """Resolve a configuration value by precedence."""
    if override is not None:
        return override
    if env_value is not None:
        return env_value
    if config_value is not None:
        return config_value
    return default_value


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect whether an exception was caused by a 429 rate limit."""
    status_candidates = [
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(exc, "http_status", None),
    ]
    response = getattr(exc, "response", None)
    if response is not None:
        status_candidates.extend(
            [
                getattr(response, "status_code", None),
                getattr(response, "status", None),
            ]
        )
    if any(code == 429 for code in status_candidates if code is not None):
        return True
    message = str(exc).lower()
    return "rate limit" in message or "too many requests" in message or "status code: 429" in message


def _get_status_code(exc: Exception) -> int | None:
    """Extract an HTTP-like status code from common exception shapes."""

    status_candidates = [
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(exc, "http_status", None),
    ]
    response = getattr(exc, "response", None)
    if response is not None:
        status_candidates.extend(
            [
                getattr(response, "status_code", None),
                getattr(response, "status", None),
            ]
        )
    for code in status_candidates:
        if isinstance(code, int):
            return code
    return None


def _is_response_format_error(exc: Exception) -> bool:
    """Return True when the provider rejects response_format usage."""

    status = _get_status_code(exc)
    message = str(exc).lower()
    return status == 400 and "response_format" in message and (
        "not supported" in message or "unsupported" in message or "invalid" in message
    )


def _is_non_retriable_error(exc: Exception) -> bool:
    """Detect failures that should not be retried."""

    status = _get_status_code(exc)
    if status in {400, 401, 403} and not _is_rate_limit_error(exc):
        return True

    message = str(exc).lower()
    fatal_markers = [
        "invalid api key",
        "incorrect api key",
        "unknown model",
        "unsupported model",
        "access was denied",
        "authentication error",
        "invalid request",
    ]
    return any(marker in message for marker in fatal_markers)


def load_api_key(
    api_key: str | None = None,
    api_key_file: str | Path | None = None,
) -> str:
    """Load the API key from parameters, environment, or file.

    Args:
        api_key: Direct API key string (highest precedence).
        api_key_file: Path to the API key file.

    Returns:
        The resolved API key string.

    Raises:
        APIKeyError: If the key cannot be located or is empty.
    """
    if api_key:
        return api_key

    env_key = os.environ.get(ENV_API_KEY)
    if env_key:
        return env_key

    if api_key_file is None:
        api_key_file = os.environ.get(ENV_KEY_FILE, DEFAULT_KEY_FILE)

    key_path = Path(api_key_file)
    if not key_path.exists():
        raise APIKeyError(
            f"API key file not found at '{key_path}'. "
            f"Set {ENV_API_KEY} or provide a valid key file path."
        )

    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise APIKeyError(f"API key file '{key_path}' is empty.")

    return key


def _truncate(text: str, limit: int = 500) -> str:
    """Truncate long text for logging."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated]"


def _join_content_parts(message_content: Any) -> str:
    """Normalize mixed message content (list parts vs string) into a string."""
    if message_content is None:
        raise ValueError("Empty response content from model")

    if isinstance(message_content, list):
        parts: List[str] = []
        for part in message_content:
            text_val = None
            if isinstance(part, dict):
                text_val = part.get("text")
            else:
                text_val = getattr(part, "text", None)
            if text_val:
                parts.append(str(text_val))
            else:
                parts.append(str(part))
        return "".join(parts)

    return str(message_content)


def _coerce_message_text(message_content: Any) -> str:
    """
    Normalize the message content returned by the LLM to a plain string.

    Models like DeepSeek can return content parts (list objects) or prepend
    `<think> ... </think>` reasoning blocks that break JSON parsing. This helper
    joins content parts, strips reasoning blocks, and validates that something
    usable remains.
    """
    message_text = _join_content_parts(message_content)

    stripped = re.sub(r"<think>.*?</think>", "", message_text, flags=re.DOTALL)
    removed_len = len(message_text) - len(stripped)
    cleaned = stripped.strip()
    if removed_len > 0:
        logger.debug("Stripped <think> block(s) from response content (removed %s chars)", removed_len)
    if not cleaned:
        raise ValueError("Empty response content from model after cleaning")
    return cleaned


def _normalize_content_and_reasoning(
    message_content: Any,
    *,
    uses_think_tags: bool,
    capture_reasoning: bool,
) -> Tuple[str, Optional[str]]:
    """
    Coerce message content to string and optionally capture reasoning segments.

    Args:
        message_content: Raw message content from the SDK.
        uses_think_tags: Whether the model nests reasoning inside <think> tags.
        capture_reasoning: Whether to return reasoning content if present.

    Returns:
        Tuple of (cleaned_content, reasoning_content or None).
    """
    message_text = _join_content_parts(message_content)
    reasoning_content: Optional[str] = None

    if uses_think_tags:
        think_blocks = re.findall(r"<think>(.*?)</think>", message_text, flags=re.DOTALL)
        if think_blocks:
            removed_chars = sum(len(block) for block in think_blocks)
            logger.debug(
                "Stripping %s <think> block(s) totaling %s characters",
                len(think_blocks),
                removed_chars,
            )
            if capture_reasoning:
                reasoning_segments = [block.strip() for block in think_blocks if block.strip()]
                if reasoning_segments:
                    reasoning_content = "\n\n".join(reasoning_segments)
        message_text = re.sub(r"<think>.*?</think>", "", message_text, flags=re.DOTALL)

    cleaned = message_text.strip()
    if not cleaned:
        raise ResponseParseError("Empty response content from model after cleaning")
    return cleaned, reasoning_content


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Find the first balanced JSON object in a text blob.

    Useful when a model ignores response_format and wraps the JSON in prose
    (common with some DeepSeek variants).
    """
    depth = 0
    start_idx: Optional[int] = None
    for idx, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    logger.debug("Extracted balanced JSON object ending at index %s", idx)
                    return text[start_idx : idx + 1]
    return None


def _strip_markdown_code_fences(text: str) -> str:
    """Remove ``` or ```json fences that models sometimes add around JSON."""
    stripped = text.strip()
    fenced = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced)
    if fenced != stripped:
        logger.debug("Removed markdown code fences from response content")
    return fenced.strip()


def _load_json_lenient(text: str) -> Any:
    """
    Try to parse JSON while tolerating common LLM formatting issues:
    - Surrounding ```/```json fences
    - Trailing commas before } or ]
    """
    cleaned = _strip_markdown_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception:
        cleaned_no_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        if cleaned_no_trailing_commas != cleaned:
            logger.debug("Removed trailing commas before JSON parse attempt")
            return json.loads(cleaned_no_trailing_commas)
        raise


class ModelClient:
    """A configurable client for OpenAI-compatible LLM APIs.

    Handles API key resolution, retry logic with exponential backoff,
    and JSON parsing retries for structured responses.
    """

    def __init__(
        self,
        model: str | None = None,
        api_base_url: str | None = None,
        api_key: str | None = None,
        api_key_file: str | Path | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        use_reasoning_model: bool | None = None,
        capture_reasoning: bool | None = None,
        max_retries: int | None = None,
        retry_base_delay: float | None = None,
        parse_retries: int | None = None,
        parse_retry_delay: float | None = None,
        request_timeout: float | None = None,
        response_format: Dict[str, Any] | None = None,
        config: ModelClientConfig | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        config_obj = config or ModelClientConfig()
        file_config = _load_config_file(config_path)

        self.model = _resolve_value(
            model,
            os.environ.get(ENV_MODEL),
            file_config.get("model", config_obj.model),
            DEFAULT_MODEL,
        )
        self.model_spec = MODEL_REGISTRY.get(self.model) or ModelSpec(
            identifier=self.model,
            model_type=ModelType.STANDARD,
            default_temperature=DEFAULT_TEMPERATURE,
        )
        self.api_base_url = _resolve_value(
            api_base_url,
            os.environ.get(ENV_API_BASE_URL),
            file_config.get("api_base_url", config_obj.api_base_url),
            DEFAULT_API_BASE_URL,
        )
        self.temperature = float(
            _resolve_value(
                temperature,
                None,
                file_config.get("temperature", config_obj.temperature),
                DEFAULT_TEMPERATURE,
            )
        )
        if (
            temperature is None
            and "temperature" not in file_config
            and config is None
            and self.model_spec.default_temperature is not None
            and self.model_spec.model_type == ModelType.REASONING
        ):
            self.temperature = float(self.model_spec.default_temperature)
        self.top_p = _resolve_value(
            top_p,
            None,
            file_config.get("top_p", config_obj.top_p),
            self.model_spec.default_top_p,
        )
        self.max_tokens = _resolve_value(
            max_tokens,
            None,
            file_config.get("max_tokens", config_obj.max_tokens),
            self.model_spec.default_max_tokens,
        )
        cfg_use_reasoning = file_config.get("use_reasoning_model")
        cfg_capture_reasoning = file_config.get("capture_reasoning")

        use_reasoning_val = _resolve_value(
            use_reasoning_model,
            None,
            cfg_use_reasoning,
            config_obj.use_reasoning_model,
        )
        if use_reasoning_val is None:
            use_reasoning_val = self.model_spec.model_type == ModelType.REASONING
        self.use_reasoning_model = bool(use_reasoning_val)

        capture_val = _resolve_value(
            capture_reasoning,
            None,
            cfg_capture_reasoning,
            config_obj.capture_reasoning,
        )
        if capture_val is None:
            capture_val = self.use_reasoning_model
        self.capture_reasoning = bool(capture_val)
        self.max_retries = int(
            _resolve_value(
                max_retries,
                None,
                file_config.get("max_retries", config_obj.max_retries),
                DEFAULT_MAX_RETRIES,
            )
        )
        self.retry_base_delay = float(
            _resolve_value(
                retry_base_delay,
                None,
                file_config.get("retry_base_delay", config_obj.retry_base_delay),
                DEFAULT_RETRY_BASE_DELAY,
            )
        )
        self.parse_retries = int(
            _resolve_value(
                parse_retries,
                None,
                file_config.get("parse_retries", config_obj.parse_retries),
                DEFAULT_PARSE_RETRIES,
            )
        )
        self.parse_retry_delay = float(
            _resolve_value(
                parse_retry_delay,
                None,
                file_config.get("parse_retry_delay", config_obj.parse_retry_delay),
                DEFAULT_PARSE_RETRY_DELAY,
            )
        )
        req_timeout_value = _resolve_value(
            request_timeout,
            None,
            file_config.get("request_timeout", config_obj.request_timeout),
            DEFAULT_REQUEST_TIMEOUT,
        )
        self.request_timeout = float(req_timeout_value) if req_timeout_value is not None else None
        response_fmt = (
            response_format
            if response_format is not None
            else file_config.get("response_format", config_obj.response_format)
        )
        if response_fmt is None:
            response_fmt = DEFAULT_RESPONSE_FORMAT
        self._response_format: Dict[str, Any] = dict(response_fmt)

        resolved_key_file = _resolve_value(
            api_key_file,
            os.environ.get(ENV_KEY_FILE),
            file_config.get("api_key_file", config_obj.api_key_file),
            DEFAULT_KEY_FILE,
        )
        resolved_api_key = _resolve_value(
            api_key,
            os.environ.get(ENV_API_KEY),
            file_config.get("api_key", config_obj.api_key),
            None,
        )
        api_key_value = load_api_key(api_key=resolved_api_key, api_key_file=resolved_key_file)

        self._client = OpenAI(api_key=api_key_value, base_url=self.api_base_url)

        logger.debug(
            "ModelClient initialized with model=%s, base_url=%s, temperature=%s, max_retries=%s",
            self.model,
            self.api_base_url,
            self.temperature,
            self.max_retries,
        )

    def _chat_completion_with_retry(
        self,
        system_prompt: str,
        user_content: str,
        response_format: Dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        use_default_format: bool = True,
        allow_format_fallback: bool = True,
    ):
        """Make a chat completion request with retry logic, returning the message object."""
        last_error: Exception | None = None
        temp = temperature if temperature is not None else self.temperature
        fmt = (
            response_format
            if (response_format is not None or not use_default_format)
            else self._response_format
        )
        nucleus = top_p if top_p is not None else self.top_p
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        attempt = 0
        max_attempts = self.max_retries
        while attempt < max_attempts:
            try:
                request_params: Dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": temp,
                    "timeout": self.request_timeout,
                }
                if fmt is not None:
                    request_params["response_format"] = fmt
                if nucleus is not None:
                    request_params["top_p"] = nucleus
                if max_tok is not None:
                    request_params["max_tokens"] = max_tok

                response = self._client.chat.completions.create(**request_params)
                message = response.choices[0].message
                content_preview = getattr(message, "content", None)
                preview_text = _truncate(str(content_preview)) if content_preview is not None else ""
                logger.debug(
                    "Model call succeeded (attempt %s): %s",
                    attempt + 1,
                    preview_text,
                )
                if getattr(message, "parsed", None) is None and not content_preview:
                    raise ModelCallError("Empty response content from model.")
                return message
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if allow_format_fallback and fmt is not None and _is_response_format_error(exc):
                    logger.warning("response_format rejected by provider; retrying without response_format.")
                    fmt = None
                    allow_format_fallback = False
                    attempt = 0
                    max_attempts = self.max_retries
                    last_error = None
                    continue

                if _is_non_retriable_error(exc):
                    raise ModelCallError(f"Non-retriable error from model API: {exc}") from exc

                attempt += 1
                rate_limited = _is_rate_limit_error(exc)
                if rate_limited and max_attempts < RATE_LIMIT_MAX_ATTEMPTS:
                    max_attempts = RATE_LIMIT_MAX_ATTEMPTS
                if attempt >= max_attempts:
                    break
                if rate_limited:
                    delay_index = min(attempt - 1, len(RATE_LIMIT_BACKOFF_SECONDS) - 1)
                    delay = RATE_LIMIT_BACKOFF_SECONDS[delay_index]
                else:
                    delay = self.retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "API call failed%s (attempt %s/%s): %s. Retrying in %.1fs",
                    " due to rate limit (429)" if rate_limited else "",
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise ModelCallError(
            f"Model call failed after {max_attempts} attempts: {last_error}"
        ) from last_error

    def _call_api(
        self,
        system_prompt: str,
        user_content: str,
        response_format: Dict[str, Any] | None = None,
        temperature: float | None = None,
        use_default_response_format: bool = True,
    ) -> str:
        """Make an API call with retry logic.

        Args:
            system_prompt: System prompt text.
            user_content: User message content.
            response_format: Response format to request.
            temperature: Optional override temperature.

        Returns:
            Raw string content from the model response.

        Raises:
            ModelCallError: If the call fails after all retries.
        """
        message = self._chat_completion_with_retry(
            system_prompt=system_prompt,
            user_content=user_content,
            response_format=response_format,
            temperature=temperature,
            use_default_format=use_default_response_format,
        )
        content = message.content
        if not content:
            raise ModelCallError("Empty response content from model.")
        return _join_content_parts(content)

    def complete(
        self,
        system_prompt: str,
        user_content: str,
        response_format: Dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a completion request and return the raw response content.

        Args:
            system_prompt: System prompt text.
            user_content: User message content.
            response_format: Optional response format override.
            temperature: Optional temperature override.

        Returns:
            Raw string content from the model.

        Raises:
            ModelCallError: If the call fails after retries.
        """
        return self._call_api(
            system_prompt=system_prompt,
            user_content=user_content,
            response_format=response_format,
            temperature=temperature,
            use_default_response_format=False,
        )

    def complete_json_with_reasoning(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        capture_reasoning: bool | None = None,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Send a completion request expecting JSON and optionally capture reasoning.

        Args:
            system_prompt: System prompt text.
            user_content: User message content.
            temperature: Optional temperature override.
            top_p: Optional nucleus sampling override.
            max_tokens: Optional max_tokens override.
            capture_reasoning: Override whether to return reasoning content.

        Returns:
            Tuple of (parsed JSON payload, reasoning content or None).

        Raises:
            ModelCallError: If the API call fails after retries.
            ResponseParseError: If the response cannot be parsed as JSON.
        """
        json_format = {"type": "json_object"}
        last_error: Exception | None = None
        last_content_preview: str | None = None
        capture = self.capture_reasoning if capture_reasoning is None else capture_reasoning
        if capture_reasoning is None and not capture and self.use_reasoning_model:
            # Default to capturing reasoning when using reasoning-capable models.
            capture = True
        nucleus = top_p if top_p is not None else self.top_p
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        uses_think_tags = self.model_spec.uses_think_tags
        supports_reasoning_attr = self.model_spec.supports_reasoning_content

        for parse_attempt in range(self.parse_retries):
            try:
                message = self._chat_completion_with_retry(
                    system_prompt=system_prompt,
                    user_content=user_content,
                    response_format=json_format,
                    temperature=temperature,
                    top_p=nucleus,
                    max_tokens=max_tok,
                )
                message_content = message.content
                parsed_payload = getattr(message, "parsed", None)
                reasoning_payload: Optional[str] = None
                preview_source = _join_content_parts(message_content) if message_content is not None else ""
                last_content_preview = _truncate(preview_source)

                if parsed_payload is None:
                    normalized_content, think_reasoning = _normalize_content_and_reasoning(
                        message_content,
                        uses_think_tags=uses_think_tags,
                        capture_reasoning=capture,
                    )
                    try:
                        parsed_payload = _load_json_lenient(normalized_content)
                    except Exception as exc:  # noqa: BLE001
                        json_fragment = _extract_first_json_object(normalized_content)
                        if json_fragment:
                            logger.debug("Falling back to first JSON object extraction.")
                            parsed_payload = _load_json_lenient(json_fragment)
                        else:
                            raise ResponseParseError(
                                "Model response was not valid JSON; preview: "
                                f"{_truncate(normalized_content)}"
                            ) from exc
                    reasoning_payload = think_reasoning

                if self.use_reasoning_model and capture:
                    if supports_reasoning_attr:
                        attr_reasoning = getattr(message, "reasoning_content", None)
                        if attr_reasoning:
                            reasoning_payload = attr_reasoning
                    elif uses_think_tags and reasoning_payload is None and message_content is not None:
                        # If parsed came from SDK parsed field we still want think reasoning
                        _, reasoning_payload = _normalize_content_and_reasoning(
                            message_content,
                            uses_think_tags=uses_think_tags,
                            capture_reasoning=True,
                        )

                if parsed_payload is None:
                    raise ResponseParseError("Failed to parse model response as JSON")

                return parsed_payload, reasoning_payload if capture else None
            except ModelCallError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if parse_attempt < self.parse_retries - 1:
                    logger.warning(
                        "JSON parse failed (attempt %s/%s): %s. Retrying... Response preview: %s",
                        parse_attempt + 1,
                        self.parse_retries,
                        exc,
                        last_content_preview,
                    )
                    time.sleep(self.parse_retry_delay)
                    continue
                raise ResponseParseError(
                    "Failed to parse model response as JSON after "
                    f"{self.parse_retries} attempts: {exc}. Response preview: {last_content_preview}"
                ) from exc

        raise ResponseParseError(
            f"Exhausted JSON parsing attempts: {last_error}. Last response preview: {last_content_preview}"
        ) from last_error

    def complete_json(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float | None = None,
    ) -> Dict[str, Any]:
        """Backward-compatible JSON completion helper."""
        result, _ = self.complete_json_with_reasoning(
            system_prompt=system_prompt,
            user_content=user_content,
            temperature=temperature,
        )
        return result

    @property
    def client(self) -> OpenAI:
        """Access the underlying OpenAI client for advanced usage."""
        return self._client


_default_client: Optional[ModelClient] = None


def get_default_client() -> ModelClient:
    """Get or create the default ModelClient instance.

    Returns:
        A ModelClient configured via environment variables and defaults.
    """
    global _default_client
    if _default_client is None:
        _default_client = ModelClient()
    return _default_client


def _get_client_for(model: str | None = None) -> ModelClient:
    """Return a ModelClient using the provided model or the shared default."""
    if model is None:
        return get_default_client()
    return ModelClient(model=model)


def complete_json(
    system_prompt: str,
    user_content: str,
    temperature: float | None = None,
    model: str | None = None,
) -> Dict[str, Any]:
    """Convenience function to make a JSON completion request.

    Args:
        system_prompt: System prompt text.
        user_content: User message content.
        temperature: Optional temperature override.
        model: Optional model override for this call.

    Returns:
        Parsed JSON response.

    Raises:
        ModelCallError: If the API call fails.
        ResponseParseError: If JSON parsing fails.
    """
    client = _get_client_for(model=model)
    return client.complete_json(
        system_prompt=system_prompt,
        user_content=user_content,
        temperature=temperature,
    )


def complete_json_with_reasoning(
    system_prompt: str,
    user_content: str,
    temperature: float | None = None,
    model: str | None = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Convenience function to make a JSON completion request with reasoning."""
    client = _get_client_for(model=model)
    return client.complete_json_with_reasoning(
        system_prompt=system_prompt,
        user_content=user_content,
        temperature=temperature,
    )

