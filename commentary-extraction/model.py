"""
Lightweight interface to OpenAI's Responses API for commentary extraction.

This module:
- Reads the OpenAI API key from `keys/openai/key` (plain text).
- Sends requests to the Responses API using the GPT-5.2 model with optional Extra High reasoning effort (enabled by default).
- Structures requests to maximize prompt cache hit rates (static content first, dynamic content last).
- Supports structured outputs for guaranteed JSON schema compliance.

PROMPT CACHING:
    Prompt caching is automatic and reduces latency by up to 80% and input costs by 50%.
    To benefit from caching, ensure your system_prompt is at least 1,024 tokens.
    Always place static/stable content in system_prompt and variable content in user_message.

STRUCTURED OUTPUTS:
    Use query_structured() or query_structured_with_metadata() to constrain the model's
    output to a specific JSON schema. This guarantees valid, parseable JSON responses
    that conform exactly to your specified structure. Structured calls use the
    Responses API's `text.format=json_schema` payload shape (the legacy
    `response_format` field is rejected) and must include both a `schema` property at
    `text.format.schema` and a `name` property at `text.format.name`.

Functions:
    query(system_prompt, user_message) -> str
        Basic query, returns response text.
    
    query_with_metadata(system_prompt, user_message) -> dict
        Returns response text plus usage/caching metadata.
    
    query_structured(system_prompt, user_message, response_schema) -> dict
        Returns parsed JSON conforming to the provided schema.
    
    query_structured_with_metadata(system_prompt, user_message, response_schema) -> dict
        Returns parsed JSON plus usage/caching metadata.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict

import requests

API_KEY_PATH = os.path.join("keys", "openai", "key")
RESPONSES_URL = "https://api.openai.com/v1/responses"
MODEL_NAME = "gpt-5.2"
REASONING_EFFORT = "xhigh"  # Extra High
RETRY_BACKOFF_SECONDS = [2, 4, 8, 16, 32]

# Prompt caching activates automatically when the prompt prefix is at least
# 1,024 tokens. Ensure your system_prompt is sufficiently long to benefit
# from cached input tokens (50% cost reduction, up to 80% latency reduction).
MIN_TOKENS_FOR_CACHE = 1024
OUTPUT_TEXT_TYPES = {"text", "output_text", "summary_text"}

logger = logging.getLogger(__name__)


def _read_api_key(path: str = API_KEY_PATH) -> str:
    """
    Load the API key from disk.

    Raises:
        FileNotFoundError: if the key file does not exist.
        ValueError: if the key file is empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"API key file not found at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        key = fh.read().strip()
    if not key:
        raise ValueError(f"API key file at {path} is empty")
    return key


def _build_payload(
    system_prompt: str,
    user_message: str,
    *,
    use_reasoning: bool = True,
    reasoning_effort: str = REASONING_EFFORT,
) -> Dict[str, Any]:
    """
    Build a Responses API payload that keeps the static system prompt first to
    maximize prompt-cache reuse.

    Uses `input_text` content blocks (the Responses API rejects the legacy
    `text` type with an invalid_value error). Set use_reasoning to False to omit
    the reasoning configuration.
    """
    payload = {
        "model": MODEL_NAME,
        # Responses API uses `input` (array of message objects) in current docs.
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_message}],
            },
        ],
        # Prompt caching is automatic for supported models; keeping static
        # content first increases cache hit rate. No explicit toggle needed.
    }
    if use_reasoning:
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def _build_structured_payload(
    system_prompt: str,
    user_message: str,
    response_schema: Dict[str, Any],
    schema_name: str = "structured_response",
    *,
    use_reasoning: bool = True,
    reasoning_effort: str = REASONING_EFFORT,
) -> Dict[str, Any]:
    """
    Build a Responses API payload with structured output constraints.

    The schema must be a valid JSON Schema object. All fields should be
    marked as required, and additionalProperties should be false for
    strict mode compatibility.

    Uses `input_text` content blocks and the current `text.format`
    field expected by the Responses API for JSON Schema outputs.
    The `schema` key at `text.format.schema` and `name` key at
    `text.format.name` are required by the API and also used for
    schema caching. Set use_reasoning to False to omit the reasoning
    configuration.
    """
    payload = {
        "model": MODEL_NAME,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_message}]},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": response_schema,
                "strict": True,
            },
        },
    }
    if use_reasoning:
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def _extract_output_text(resp: Dict[str, Any]) -> str:
    """
    Extract the model's text output from a Responses API payload.
    Handles common shapes used by the API.
    """
    if "output_text" in resp and isinstance(resp["output_text"], str):
        return resp["output_text"]

    if "output" in resp:
        output = resp["output"]
        if isinstance(output, str):
            return output
        if isinstance(output, dict) and isinstance(output.get("text"), str):
            return output["text"]
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if "parsed" in item:
                    parsed_val = item["parsed"]
                    try:
                        return json.dumps(parsed_val)
                    except TypeError:
                        return str(parsed_val)
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        part_type = part.get("type")
                        if part_type in OUTPUT_TEXT_TYPES and isinstance(part.get("text"), str):
                            return part["text"]
                        if "parsed" in part:
                            parsed_val = part["parsed"]
                            try:
                                return json.dumps(parsed_val)
                            except TypeError:
                                return str(parsed_val)
                if isinstance(item.get("text"), str):
                    return item["text"]

    if "choices" in resp and resp["choices"]:
        choice = resp["choices"][0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(choice.get("text"), str):
                return choice["text"]

    raise ValueError("Could not extract text from API response")


def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the HTTP POST to the Responses API with retries and return the parsed JSON body.
    """
    api_key = _read_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    max_attempts = len(RETRY_BACKOFF_SECONDS) + 1

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(
                RESPONSES_URL, headers=headers, data=json.dumps(payload), timeout=6000
            )
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                raise requests.HTTPError(f"{exc}: {resp.text}") from exc

            try:
                result = resp.json()
            except ValueError as exc:
                raise ValueError(f"Failed to decode JSON response: {resp.text}") from exc

            if attempt > 1:
                logger.info("API call succeeded on attempt %s/%s", attempt, max_attempts)
            return result
        except Exception as exc:  # Retry on any failure within the call
            if attempt >= max_attempts:
                logger.error(
                    "API call failed after %s attempts. Last error: %s. Terminating.",
                    attempt,
                    exc,
                )
                sys.exit(1)

            delay = RETRY_BACKOFF_SECONDS[attempt - 1]
            logger.warning(
                "API call failed (attempt %s/%s): %s. Retrying in %ss...",
                attempt,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)


def query(
    system_prompt: str,
    user_message: str,
    *,
    use_reasoning: bool = True,
    reasoning_effort: str = REASONING_EFFORT,
) -> str:
    """
    Send a query to GPT-5.2 (Responses API) with optional Extra High reasoning effort.

    Args:
        system_prompt: Stable system instructions (put long/static content here for caching).
        user_message: The user's question or content to process.
        use_reasoning: Include reasoning configuration when True (default).

    Returns:
        The response text from the model.
    """
    payload = _build_payload(
        system_prompt, user_message, use_reasoning=use_reasoning, reasoning_effort=reasoning_effort
    )
    response_body = _post(payload)
    err = response_body.get("error")
    if err:
        raise ValueError(f"API error: {err}")
    return _extract_output_text(response_body)


def query_with_metadata(
    system_prompt: str,
    user_message: str,
    *,
    use_reasoning: bool = True,
    reasoning_effort: str = REASONING_EFFORT,
) -> Dict[str, Any]:
    """
    Variant of `query` that also returns usage/caching metadata when available.

    Returns:
        A dict with:
            - text: response text
            - usage: usage object from API (may include cached_tokens)
            - raw: full parsed response for debugging
    """
    payload = _build_payload(
        system_prompt, user_message, use_reasoning=use_reasoning, reasoning_effort=reasoning_effort
    )
    response_body = _post(payload)
    err = response_body.get("error")
    if err:
        raise ValueError(f"API error: {err}")
    return {
        "text": _extract_output_text(response_body),
        "usage": response_body.get("usage"),
        "raw": response_body,
    }


def query_structured(
    system_prompt: str,
    user_message: str,
    response_schema: Dict[str, Any],
    schema_name: str = "structured_response",
    *,
    use_reasoning: bool = True,
    reasoning_effort: str = REASONING_EFFORT,
) -> Dict[str, Any]:
    """
    Send a query to GPT-5.2 with structured output constraints and optional reasoning.

    The model's response is guaranteed to be valid JSON conforming to
    the provided schema. This eliminates parsing failures and ensures
    all required fields are present.

    Args:
        system_prompt: Stable system instructions (static content for caching).
        user_message: The user's question or content to process.
        response_schema: A JSON Schema dict defining the expected output structure.
        schema_name: Name for the schema (used for schema caching).
        use_reasoning: Include reasoning configuration when True (default).

    Returns:
        The parsed JSON object from the model's response.
    """
    payload = _build_structured_payload(
        system_prompt,
        user_message,
        response_schema,
        schema_name,
        use_reasoning=use_reasoning,
        reasoning_effort=reasoning_effort,
    )
    response_body = _post(payload)
    err = response_body.get("error")
    if err:
        raise ValueError(f"API error: {err}")
    text_output = _extract_output_text(response_body)
    return json.loads(text_output)


def query_structured_with_metadata(
    system_prompt: str,
    user_message: str,
    response_schema: Dict[str, Any],
    schema_name: str = "structured_response",
    *,
    use_reasoning: bool = True,
    reasoning_effort: str = REASONING_EFFORT,
) -> Dict[str, Any]:
    """
    Variant of query_structured that also returns usage/caching metadata.

    Returns:
        A dict with:
            - data: the parsed JSON object from the response
            - usage: usage object from API (includes cached_tokens when applicable)
            - raw: full parsed response for debugging
    """
    payload = _build_structured_payload(
        system_prompt,
        user_message,
        response_schema,
        schema_name,
        use_reasoning=use_reasoning,
        reasoning_effort=reasoning_effort,
    )
    response_body = _post(payload)
    err = response_body.get("error")
    if err:
        raise ValueError(f"API error: {err}")
    text_output = _extract_output_text(response_body)
    return {
        "data": json.loads(text_output),
        "usage": response_body.get("usage"),
        "raw": response_body,
    }


__all__ = [
    "query",
    "query_with_metadata",
    "query_structured",
    "query_structured_with_metadata",
]

