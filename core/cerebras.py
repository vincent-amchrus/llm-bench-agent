# core/chat_client.py
import time
import json
from typing import List, Dict, Optional, Union
from openai import OpenAI, AsyncOpenAI
import toon_format



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