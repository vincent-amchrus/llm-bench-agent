# # config/tools.py
# WEATHER_TOOL = {
#     "type": "function",
#     "function": {
#         "name": "weather_tool",
#         "description": "Get weather-related info (UV, temp, etc.)",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "rewrite_message": {
#                     "type": "string",
#                     "description": "User query paraphrased for clarity",
#                     "match_mode": "semantic"  # 🔴 NEW: specify match strategy
#                 },
#                 "time": {
#                     "type": "string",
#                     "description": "Time in HH:MM or '2am' format",
#                     "match_mode": "time"  # parsed & compared as datetime
#                 }
#             },
#             "required": ["rewrite_message"]
#         }
#     }
# }

# VEHICLE_TOOL = {
#     "type": "function",
#     "function": {
#         "name": "vehicle_control",
#         "description": "Control vehicle features",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "object": {
#                     "type": "string",
#                     "enum": ["VOLUME", "LIGHTS", "AC"],
#                     "match_mode": "exact"  # enum → exact
#                 },
#                 "action": {
#                     "type": "string",
#                     "enum": ["increase", "decrease", "toggle"],
#                     "match_mode": "exact"
#                 },
#                 "value": {
#                     "type": "integer",
#                     "match_mode": "exact"  # numeric
#                 }
#             },
#             "required": ["object", "action"]
#         }
#     }
# }

# ALL_TOOLS = [WEATHER_TOOL, VEHICLE_TOOL]


import json
ALL_TOOLS = json.load(open("data/tools/vivi_global_tools.json"))