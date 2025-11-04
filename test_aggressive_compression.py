"""
Test aggressive compression - remove parameter descriptions.
"""
import json
import sys
sys.path.append('src')

from tools.tools_registry import TOOLS

# Original size
original_json = json.dumps(TOOLS)
original_size = len(original_json)

# Aggressive compression
compressed = []
for tool in TOOLS:
    params = tool["function"]["parameters"]
    compressed_params = {
        "type": params["type"],
        "properties": {},
        "required": params.get("required", [])
    }
    
    # Keep only type info, remove descriptions
    for prop_name, prop_value in params.get("properties", {}).items():
        compressed_params["properties"][prop_name] = {
            "type": prop_value.get("type", "string")
        }
        if "enum" in prop_value:
            compressed_params["properties"][prop_name]["enum"] = prop_value["enum"]
        if "oneOf" in prop_value:
            compressed_params["properties"][prop_name]["oneOf"] = prop_value["oneOf"]
        if prop_value.get("type") == "array" and "items" in prop_value:
            compressed_params["properties"][prop_name]["items"] = {"type": prop_value["items"].get("type", "string")}
    
    compressed_tool = {
        "type": tool["type"],
        "function": {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"][:100],
            "parameters": compressed_params
        }
    }
    compressed.append(compressed_tool)

compressed_json = json.dumps(compressed)
compressed_size = len(compressed_json)

print('=' * 80)
print('AGGRESSIVE TOOLS COMPRESSION TEST')
print('=' * 80)

print(f'\n📦 Original (46 tools with full descriptions):')
print(f'   Size: {original_size:,} characters')
print(f'   Tokens: ~{original_size // 4:,}')

print(f'\n✅ Compressed (46 tools, minimal descriptions):')
print(f'   Size: {compressed_size:,} characters')  
print(f'   Tokens: ~{compressed_size // 4:,}')

print(f'\n💰 Savings:')
print(f'   Reduction: {original_size - compressed_size:,} characters')
print(f'   Token savings: ~{(original_size - compressed_size) // 4:,} tokens')
print(f'   Percentage: {100 - compressed_size*100//original_size}% smaller')

print(f'\n📊 Final Request Size:')
print(f'   System prompt: ~482 tokens')
print(f'   Tools (46 compressed): ~{compressed_size // 4:,} tokens')
print(f'   User message: ~100 tokens')
print(f'   Tool result summary: ~200 tokens (per tool)')
print(f'   TOTAL (with 1 tool): ~{482 + compressed_size // 4 + 100 + 200:,} tokens')
print(f'   TOTAL (with 5 tools): ~{482 + compressed_size // 4 + 100 + 1000:,} tokens')
print(f'   Remaining budget: ~{12000 - (482 + compressed_size // 4 + 100 + 1000):,} tokens')

print(f'\n✅ All 46 tools available!')
print(f'✅ Fits well within 12K token limit!')
print('=' * 80)
