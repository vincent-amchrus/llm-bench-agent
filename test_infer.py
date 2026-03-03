from openai import OpenAI

BASE_URL="http://localhost:8268/v1"
MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"

MODEL = "unsloth/Qwen3-4B-Instruct-2507"
MODEL = "ckp_8562_fc_29k_smart_39k_global"


# BASE_URL="http://100.96.10.15:8006/v1"
# BASE_URL = "http://10.149.3.12:8006/v1"
BASE_URL = "http://localhost:8006/v1"
API_KEY="token-vllm"
MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"

# # # BASE_URL = "http://localhost:8269/v1"
# # BASE_URL = "http://localhost:8269/v1"
# # BASE_URL = "http://localhost:8265/v1"
# BASE_URL = "http://localhost:8269/v1"
BASE_URL = "http://localhost:8268/v1"
API_KEY="EMPTY"

client = OpenAI(
    base_url = BASE_URL,
    api_key = API_KEY
)
from pprint import pprint
for model in client.models.list():
    print(model.id)


# client.chat.completions.create(
#     model='Qwen/Qwen3-Next-80B-A3B-Instruct',
#     messages=[
#         {"role": "user", "content": "chào bạn"}
#     ]
# )


from core.chat_client import chat_completion_async

import json
global_tools = json.load(open("data/tools/vivi_global_tools2.json"))
async def main():
    messages = "gọi cho em My"


    completion = await chat_completion_async(
        messages = messages,
        base_url = BASE_URL,
        model = model.id,
        api_key = API_KEY,
        enable_thinking=True,
        # tools = temp,
        # tools = global_tools,
        # tool_choice='none'
        tools = global_tools,
        temperature=0.7, top_p=0.8, presence_penalty=1.5
        # use_toon_format=True
    )
    print(completion)


import asyncio

asyncio.run(main())