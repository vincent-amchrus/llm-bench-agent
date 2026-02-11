import asyncio


from core.chat_client import chat_completion_async
import json

async def main():
    tools = json.load(open("data/tools/vivi_global_tools2.json"))
    BASE_URL = "http://localhost:8265/v1"
    BASE_URL = "http://localhost:8388/v1"
    API_KEY = "EMPTY"
    MODEL_0 = "Qwen/Qwen3-Coder-Next"
    messages = "decrease the undercarriage light brightness by 20 levels        "

    completion = await chat_completion_async(
        messages = messages,
        base_url = BASE_URL,
        model = MODEL_0,
        api_key = API_KEY,
        tools = tools,
        # tools = tools,
        system_prompt = None,
        # use_toon_format=True
    )

    print("Answer 1: ", completion)
  

asyncio.run(main())