# core/chat_client.py
import json
from typing import List, Dict, Optional, Union
from openai import OpenAI, AsyncOpenAI


def chat_completion(
    messages: Union[str, List[Dict[str, str]]],
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    tools: Optional[List[Dict]] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Dict:
    """
    Unified chat completion interface
    - vLLM / local LLM
    - OpenAI (GPT-4.1 / 4o)
    """

    # =========================
    # Detect backend
    # =========================
    is_openai = "api.openai.com" in base_url

    # =========================
    # Normalize messages
    # =========================
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    # =========================
    # Client
    # =========================
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # =========================
    # Base kwargs (CHUNG)
    # =========================
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    # =========================
    # vLLM-only options
    # =========================
    if not is_openai:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }

    # =========================
    # Call
    # =========================
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        return {
            "error": str(e),
            "content": None,
            "tool_calls": [],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        }

    # =========================
    # Parse response
    # =========================
    msg = response.choices[0].message
    usage = getattr(response, "usage", None)

    result = {
        "content": msg.content or "",
        "tool_calls": [],
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
    }

    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}

            result["tool_calls"].append({
                "name": tc.function.name,
                "arguments": args,
            })

    return result




async def chat_completion_async(
    messages: Union[str, List[Dict[str, str]]],
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    tools: Optional[List[Dict]] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Dict:
    """
    Unified chat completion interface for tool-calling evaluation.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    }

    if tools:
        kwargs.update({"tools": tools, "tool_choice": "auto"})
    # print(kwargs)
    try:
        response = await client.chat.completions.create(**kwargs)
    except Exception as e:
        return {
            "error": str(e),
            "content": None,
            "tool_calls": [],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        }

    # Parse response
    msg = response.choices[0].message
    usage = getattr(response, "usage", None)

    result = {
        "content": msg.content or "",
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
        "tool_calls": []
    }

    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            result["tool_calls"].append({
                "name": tc.function.name,
                "arguments": args
            })

    return result