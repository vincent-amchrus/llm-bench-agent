import time
from openai import AsyncOpenAI
import json
tools = json.load(open("data/tools/vivi_smart_8tools.json"))

async def get_ttft(
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    prompt: str = "phim marvel có gì hay"
) -> float:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    start = time.perf_counter()
    
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
        tools=tools,
        stream=True
    )
    
    # Important: use async for with AsyncStream
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            print(chunk)  # Print the first token
            end = time.perf_counter()
            # return round(end - start, 4)  # TTFT in seconds
    
    return float('inf')  # No tokens generated


# Usage example
async def main():
    ttft = await get_ttft(
        "http://localhost:8268/v1",
        "Qwen/Qwen3.5-4B"
    )
    print(f"Time to First Token: {ttft}s")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())