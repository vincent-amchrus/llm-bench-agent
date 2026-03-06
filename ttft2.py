import time
from unittest import result
from openai import OpenAI

# Connect to vLLM server
client = OpenAI(base_url="http://localhost:8268/v1", api_key="EMPTY")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"],
            },
        },
    }
]
start = time.perf_counter()
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=[{"role": "user", "content": "What is the weather in Paris?"}],
    tools=tools,
    stream=True,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
)

flag = 0
for chunk in response:
    if chunk.choices:
        choice = chunk.choices[0]
        content = choice.delta.content
        tool_calls = choice.delta.tool_calls

        # 
        reasoning = getattr(choice.delta, 'reasoning', None)
        if content or tool_calls or reasoning:
            str_log = f"Received chunk:\ncontent: {content}, \ntool_calls: {tool_calls}, \nreasoning: {reasoning}"
            print(str_log)

            end = time.perf_counter()
            # print(choice)  # Print the first token or tool call or reasoning
            time_to_first_token = round(end - start, 4)
            print(f"Time to First Token/Tool Call/Reasoning: {round(time_to_first_token, 4)}s")
            break
