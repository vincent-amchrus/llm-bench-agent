from core.chat_client import chat_completion_async, chat_completion
import json

tools = json.load(open("data/tools/vivi_smart_8tools.json"))
global_tools = json.load(open("data/tools/vivi_global_tools2.json"))
from openai import OpenAI
BASE_URL = "http://localhost:8268/v1"
API_KEY="EMPTY"

client = OpenAI(
    base_url = BASE_URL,
    api_key = API_KEY
)
from pprint import pprint
for model in client.models.list():
    print(model.id)

BASE_URL = "http://localhost:8268/v1"
API_KEY="EMPTY"
system_prompt = """You are a helpful and friendly assistant. Always respond in the same language as the user's message (auto-detect and match: Vietnamese, English, or any other language).

Tool usage rules — follow these strictly:

• You may ONLY call a tool when the question clearly and directly requires information or an action that one of the AVAILABLE tools can provide.
• In ALL other situations — especially casual conversation, greetings, jokes, personal questions, small talk, opinions, feelings, or any topic not obviously needing a tool — you MUST NOT call any tool.
• Never force, guess, or invent a reason to call a tool.
• Never call a tool just because a keyword appears in the message.

When no tool is needed → answer directly with natural text. Do not output any function call.

Keep responses natural, concise and engaging in the user's language."""

system_prompt = """Respond in the same language as the user.

Call a tool ONLY when it is clearly necessary to answer correctly using one of the available tools.

Normal chat, greetings, personal questions, jokes → no tool calls. Just reply normally.

Never call a tool unnecessarily."""


MODEL="Qwen/Qwen3.5-4B"
messages = "giá xe Vinfast là bao nhiêu"
completion = chat_completion(
    messages = messages,
    base_url = BASE_URL,
    model = MODEL,
    api_key = API_KEY,
    enable_thinking=False,
    tools = tools,
    temperature=0,
    # temperature=0,
    system_prompt=system_prompt
)
print(completion)