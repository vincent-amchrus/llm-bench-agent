import json
from toon_format import encode as toon_encode

# --- Parsers ---
def parse_json(obj):
    return json.dumps(obj, ensure_ascii=False)

def parse_toon(obj):
    return toon_encode(obj)

# --- Core Functions ---
def format_tools(tools, fn_parse):
    return "\n".join(fn_parse(tool) for tool in tools)

def inject_tool_calls_into_messages(messages, fn_parse):
    """Convert messages with 'tool_calls' into messages with enriched 'content'."""
    new_messages = []
    for turn in messages:
        content = turn.get("content", "")
        tool_calls = turn.get("tool_calls", [])
        
        if tool_calls:
            tool_call_blocks = [
                f"<tool_call>\n{fn_parse(call['function'] if 'function' in call else call)}\n</tool_call>"
                for call in tool_calls
            ]
            content = (content + "\n" + "\n".join(tool_call_blocks)).strip()
        
        new_messages.append({
            "role": turn["role"],
            "content": content
        })
    return new_messages


def get_system_message_with_tools(tools, fn_parse, use_toon_format=True):
    tool_schemas = format_tools(tools, fn_parse)
        
    # Choose instruction style
    if use_toon_format:
        system_content = f"""# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_schemas}
</tools>

For each function call, return a TOON object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
name: <function-name>
arguments:
<args1>: <value1>,
<args2>: <value2>,
<args3>: <value3>,...
<tool_call>"""
    else:
        system_content = f"""# Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {tool_schemas}
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {{"name": <function-name>, "arguments": <args-json-object>}}
    </tool_call>"""
        
    return {"role": "system", "content": system_content}

def build_conversation(
    tools,
    messages,
    fn_parse,
    use_toon_format=True
):
    """
    Returns a list of messages (dicts) with:
      - A system message containing formatted tool schemas
      - All original messages with tool_calls merged into content
    """
    # Format tool definitions
    system_message = get_system_message_with_tools(tools, fn_parse, use_toon_format)
    
    
    # Transform user/assistant/tool messages
    transformed_messages = inject_tool_calls_into_messages(messages, fn_parse)
    
    # Return full conversation as list of dicts
    return [system_message] + transformed_messages

# --- Example Usage ---
if __name__ == "__main__":
    tools = [
        {
            "function": {
                'name': 'search_flights',
                'description': 'Search flight options.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'origin': {'type': 'string'},
                        'destination': {'type': 'string'},
                        'date': {'type': 'string'}
                    },
                    'required': ['origin', 'destination']
                }
            }
        },
        {
            "function": {
                'name': 'get_price',
                'description': 'Get price for a flight ID.',
                'parameters': {
                    'type': 'object',
                    'properties': {'flight_id': {'type': 'string'}},
                    'required': ['flight_id']
                }
            }
        }
    ]

    messages = [
        {'role': 'user', 'content': 'Find me the cheapest flight from Singapore to Bangkok next Friday.'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {'function': {'name': 'search_flights', 'arguments': {"origin": "Singapore", "destination": "Bangkok", "date": "2025-04-04"}}},
                {'function': {'name': 'submit_flights', 'arguments': {"form": "Hanoi", "destination": "USA", "date": "2025-04-04"}}}
            ]
        },
        {'role': 'tool', 'content': '{"flights": [{"id": "FL123"}, {"id": "FL456"}]}'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{'function': {'name': 'get_price', 'arguments': {"flight_id": "FL123"}}}]
        },
        {'role': 'tool', 'content': '{"flight_id": "FL123", "price": 120}'},
        {'role': 'assistant', 'content': 'The cheapest option is flight FL123 priced at $120.'}
    ]

    # # Get conversation in TOON format (as list of messages)
    # conv_toon = build_conversation(
    #     tools=tools,
    #     messages=messages,
    #     fn_parse=parse_toon,
    #     use_toon_format=True
    # )

    # # Get conversation in JSON format (as list of messages)
    # conv_json = build_conversation(
    #     tools=tools,
    #     messages=messages,
    #     fn_parse=parse_json,
    #     use_toon_format=False
    # )

    # # Example: print first message (system) and second (user)
    # print("TOON System Message Preview:")
    # print(conv_toon[0]["content"][:200] + "...")
    # print("\nFirst User Message:")
    # print(conv_toon[1])

    # print()
    formatted_msgs = build_conversation(tools, messages, parse_toon, use_toon_format=True)
    text = tokenizer.apply_chat_template(formatted_msgs, tokenize=False)