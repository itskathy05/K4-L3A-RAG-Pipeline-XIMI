"""
Dispatch LLM dùng chung cho task8 (fallback vectorless local) và task10
(generation). Đặt ở module riêng thay vì trong task10 để tránh vòng lặp
import: task8 cần gọi LLM, và task8 -> task10 -> task9 -> task8 sẽ là
circular import nếu dispatch nằm trong task10.

Không test nào pin vị trí hay chữ ký của các hàm trong file này.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


class LLMError(RuntimeError):
    """Bọc mọi lỗi provider để caller xử lý thống nhất."""


def _provider_and_model(model: str | None = None) -> tuple[str, str]:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    resolved_model = model or os.getenv("LLM_MODEL", "").strip()
    if not resolved_model:
        resolved_model = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash",
            "anthropic": "claude-3-5-haiku-latest",
        }.get(provider, "gpt-4o-mini")
    return provider, resolved_model


def call_text(
    system_prompt: str,
    user_message: str,
    *,
    model: str | None = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 1024,
) -> str:
    """Gọi provider được cấu hình trong .env, trả về text thuần.

    Raise LLMError khi có sự cố provider để caller quyết định fallback/refusal.
    """
    provider, resolved_model = _provider_and_model(model)
    try:
        if provider == "openai":
            return _call_openai(system_prompt, user_message, resolved_model, temperature, top_p, max_tokens)
        if provider == "gemini":
            return _call_gemini(system_prompt, user_message, resolved_model, temperature, top_p, max_tokens)
        if provider == "anthropic":
            return _call_anthropic(system_prompt, user_message, resolved_model, temperature, top_p, max_tokens)
        raise LLMError(f"Unknown LLM_PROVIDER: {provider!r}")
    except LLMError:
        raise
    except Exception as exc:  # SDK-specific errors -> uniform LLMError
        raise LLMError(f"{provider} call failed: {exc!r}") from exc


def _call_openai(system_prompt, user_message, model, temperature, top_p, max_tokens) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3, timeout=60)
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    # Reasoning models (gpt-5*, o*) từ chối temperature/top_p và dùng
    # max_completion_tokens thay vì max_tokens.
    is_reasoning_model = model.startswith(("o1", "o3", "o4", "gpt-5"))
    if is_reasoning_model:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if content is None:
        raise LLMError("OpenAI returned empty content (possibly filtered)")
    return content


def _call_gemini(system_prompt, user_message, model, temperature, top_p, max_tokens) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        ),
    )
    text = response.text or ""
    if not text:
        raise LLMError("Gemini returned empty text (possibly blocked or MAX_TOKENS)")
    return text


def _call_anthropic(system_prompt, user_message, model, temperature, top_p, max_tokens) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text:
        raise LLMError("Anthropic returned no text blocks")
    return text
