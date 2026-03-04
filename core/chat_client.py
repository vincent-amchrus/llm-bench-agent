# core/chat_client.py
import time
import json
from typing import List, Dict, Optional, Union
from urllib import response
from openai import OpenAI, AsyncOpenAI
import toon_format

from core.standardized_function_calling_messages import get_system_message_with_tools, parse_toon

def chat_completion(
    messages: Union[str, List[Dict[str, str]]],
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    tools: Optional[List[Dict]] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    use_toon_format: bool = False,
    enable_thinking: bool = False,
    **other_kwargs
) -> Dict:
    """
    Unified chat completion interface for tool-calling evaluation.
    """
    # print("Use toon format:", use_toon_format)
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    client = OpenAI(base_url=base_url, api_key=api_key)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking
            }
        }
    }

    if tools:
        # Toon format
        if use_toon_format:
            system_prompt_message = get_system_message_with_tools(tools, fn_parse=parse_toon)
            kwargs["messages"] = [system_prompt_message] + messages
        else:
            kwargs.update({"tools": tools, "tool_choice": "auto"})
    kwargs.update(other_kwargs)
    # =========================
    # print(kwargs) 
    try:
        start_time = time.perf_counter()
        response = client.chat.completions.create(**kwargs)
        end_time = time.perf_counter()
        exe_time = end_time - start_time 
    except Exception as e:
        end_time = time.perf_counter()    # <--- ADD THIS (for error case)
        exe_time = end_time - start_time 
        return {
            "error": str(e),
            "content": None,
            "reasoning": None,
            "tool_calls": [],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "throughput": {               # <--- ADD THIS
                "exe_time": exe_time,
                "output_token_per_seconds": None,
                "total_token_per_second": None
            }
        }

    # Parse response
    msg = response.choices[0].message
    usage = getattr(response, "usage", None)

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0

    # <--- ADD CALCULATION LOGIC HERE
    if exe_time > 0:
        output_token_per_seconds = completion_tokens / exe_time
        total_token_per_second = total_tokens / exe_time
    else:
        output_token_per_seconds = 0.0
        total_token_per_second = 0.0
    result = {
        "content": msg.content or "",
        "reasoning": msg.reasoning if enable_thinking else None,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
        "tool_calls": [],
        "throughput": {
            "exe_time": round(exe_time, 2),
            "output_token_per_seconds": round(output_token_per_seconds, 2),
            "total_token_per_second": round(total_token_per_second, 2)
        }
    }

    # print("Result1", result)
    if use_toon_format and "<tool_call>" in msg.content:
        content = msg.content
        tool_calls = content.split('</tool_call>\n<tool_call>')
        result.update( {
            'tool_calls': []
        })
        for tc in tool_calls:
            try:
                tc = tc.replace("<tool_call>", "").replace("</tool_call>", "").strip()
                args = toon_format.decode(tc)
            except toon_format.ToonDecodeError:
                args = {"_raw": tc}
            result["tool_calls"].append(args)
    elif msg.tool_calls:
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
    use_toon_format: bool = False,
    enable_thinking: bool = False,
    **other_kwargs
) -> Dict:
    """
    Unified chat completion interface for tool-calling evaluation.
    """
    # print("Use toon format:", use_toon_format)
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
                "enable_thinking": enable_thinking
            }
        }
    }

    if tools:
        # Toon format
        if use_toon_format:
            system_prompt_message = get_system_message_with_tools(tools, fn_parse=parse_toon)
            kwargs["messages"] = [system_prompt_message] + messages
        else:
            kwargs.update({"tools": tools, "tool_choice": "auto"})
    kwargs.update(other_kwargs)
    # =========================
    # print(kwargs) 
    try:
        start_time = time.perf_counter()
        response = await client.chat.completions.create(**kwargs)
        end_time = time.perf_counter()
        exe_time = end_time - start_time 
    except Exception as e:
        end_time = time.perf_counter()    # <--- ADD THIS (for error case)
        exe_time = end_time - start_time 
        return {
            "error": str(e),
            "content": None,
            "reasoning": None,
            "tool_calls": [],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "throughput": {               # <--- ADD THIS
                "exe_time": exe_time,
                "output_token_per_seconds": None,
                "total_token_per_second": None
            }
        }

    # Parse response
    msg = response.choices[0].message
    usage = getattr(response, "usage", None)

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0

    # <--- ADD CALCULATION LOGIC HERE
    if exe_time > 0:
        output_token_per_seconds = completion_tokens / exe_time
        total_token_per_second = total_tokens / exe_time
    else:
        output_token_per_seconds = 0.0
        total_token_per_second = 0.0
    result = {
        "content": msg.content or "",
        "reasoning": msg.reasoning if enable_thinking else None,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
        "tool_calls": [],
        "throughput": {
            "exe_time": round(exe_time, 2),
            "output_token_per_seconds": round(output_token_per_seconds, 2),
            "total_token_per_second": round(total_token_per_second, 2)
        }
    }

    # print("Result1", result)
    if use_toon_format and "<tool_call>" in msg.content:
        content = msg.content
        tool_calls = content.split('</tool_call>\n<tool_call>')
        result.update( {
            'tool_calls': []
        })
        for tc in tool_calls:
            try:
                tc = tc.replace("<tool_call>", "").replace("</tool_call>", "").strip()
                args = toon_format.decode(tc)
            except toon_format.ToonDecodeError:
                args = {"_raw": tc}
            result["tool_calls"].append(args)
    elif msg.tool_calls:
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