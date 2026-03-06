import time
from openai import OpenAI

def get_ttft(base_url: str, model: str, api_key: str = "EMPTY", prompt: str = "Hi") -> float:
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    start = time.perf_counter()
    
    # Must use stream=True to intercept the first token
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
        stream=True
    )
    
    # Iterate until the first chunk arrives
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            end = time.perf_counter()
            return round(end - start, 4)  # TTFT in seconds
            
    return float('inf') # No tokens generated

# Usage Example
ttft = get_ttft("http://localhost:8268/v1", "Qwen/Qwen3.5-4B")
print(f"Time to First Token: {ttft}s") 