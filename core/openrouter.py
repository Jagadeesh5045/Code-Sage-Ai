"""OpenRouter API client for LLM generation."""

import json
import time
import requests

import config


def chat_completion(messages, model=None, temperature=None, max_tokens=None):
    """Send a chat completion request to OpenRouter."""
    model = model or config.LLM_MODEL
    temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
    max_tokens = max_tokens or config.LLM_MAX_TOKENS

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://codesage-ai.local",
        "X-OpenRouter-Title": "CodeSage AI",
    }

    start = time.time()
    resp = requests.post(
        config.OPENROUTER_BASE_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=120,
    )
    elapsed_ms = (time.time() - start) * 1000

    if resp.status_code != 200:
        return {
            "content": f"LLM API error ({resp.status_code}): {resp.text[:300]}",
            "model": model,
            "elapsed_ms": elapsed_ms,
            "error": True,
        }

    data = resp.json()
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})

    return {
        "content": message.get("content", ""),
        "model": data.get("model", model),
        "elapsed_ms": elapsed_ms,
        "usage": data.get("usage", {}),
        "error": False,
    }


def chat_with_reasoning(messages, model=None):
    """Send a request with reasoning enabled (for models that support it)."""
    model = model or config.LLM_REASONING_MODEL

    payload = {
        "model": model,
        "messages": messages,
        "reasoning": {"enabled": True},
    }

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://codesage-ai.local",
        "X-OpenRouter-Title": "CodeSage AI",
    }

    start = time.time()
    resp = requests.post(
        config.OPENROUTER_BASE_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=180,
    )
    elapsed_ms = (time.time() - start) * 1000

    if resp.status_code != 200:
        return {
            "content": f"LLM API error ({resp.status_code}): {resp.text[:300]}",
            "reasoning": None,
            "model": model,
            "elapsed_ms": elapsed_ms,
            "error": True,
        }

    data = resp.json()
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})

    return {
        "content": message.get("content", ""),
        "reasoning": message.get("reasoning_details") or message.get("reasoning_content"),
        "model": data.get("model", model),
        "elapsed_ms": elapsed_ms,
        "usage": data.get("usage", {}),
        "error": False,
    }


def quick_llm(prompt, system_prompt=None, model=None):
    """Convenience wrapper for single-turn LLM calls."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return chat_completion(messages, model=model)
