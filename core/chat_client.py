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




async def gpt_chat_completion_async(
    messages: Union[str, List[Dict[str, str]]],
    model: str,
    api_key: Optional[str],
    system_prompt: Optional[str] = None,
    tools: Optional[List[Dict[str, str]]] = None,
    base_url: Optional[str] = None,
    **kwargs
):
    """
    Chat completion function that supports:
    - List of messages
    - Single-string task prompt

    Returns:
        {
            "content": <assistant message>,
            "usage": {
                "prompt_tokens": ...,
                "completion_tokens": ...,
                "total_tokens": ...
            }
        }
    """

    try:
        # Convert single string prompt to chat message
        if isinstance(messages, str):
            messages = [
                {"role": "user", "content": messages}
            ]

        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url  # Add base_url to client initialization if provided
        )

        # Prepare the request parameters
        params = {
            "model": model,
            "messages": messages,
        }
        
        if tools is not None:
            params["tools"] = tools
        params.update(kwargs)
        response = await client.chat.completions.create(**params)
        
        # Extract safe content
        try:
            content = response.choices[0].message.content
        except Exception:
            content = str(response)

        usage = getattr(response, "usage", None)
        usage_data = {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        }
        

        json_response = {
            "content": content,
            "usage": usage_data
        }

        if tools:
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls is not None:
                json_tool_calls = [
                    {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments)
                    }
                    for tool_call in tool_calls
                ]
                json_response["tool_calls"] = json_tool_calls
                
        return json_response
        
    except Exception as e:
        return {

            "error": str(e)
        }


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