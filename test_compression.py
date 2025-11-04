"""
Test compression strategy for tools registry.
"""
import json
import sys
sys.path.append('src')

from tools.tools_registry import TOOLS

# Original size
original = json.dumps(TOOLS)
original_size = len(original)

# Compress by truncating descriptions
compressed = []
for tool in TOOLS:
    compressed_tool = {
        "type": tool["type"],
        "function": {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"][:150],  # Truncate
            "parameters": tool["function"]["parameters"]
        }
    }
    compressed.append(compressed_tool)

compressed_json = json.dumps(compressed)
compressed_size = len(compressed_json)

print('=' * 80)
print('TOOLS REGISTRY COMPRESSION TEST')
print('=' * 80)

print(f'\n📦 Original Registry (46 tools):')
print(f'   Size: {original_size:,} characters')
print(f'   Tokens: ~{original_size // 4:,}')

print(f'\n✅ Compressed Registry (46 tools, short descriptions):')
print(f'   Size: {compressed_size:,} characters')
print(f'   Tokens: ~{compressed_size // 4:,}')

print(f'\n💰 Savings:')
print(f'   Reduction: {original_size - compressed_size:,} characters')
print(f'   Token savings: ~{(original_size - compressed_size) // 4:,} tokens')
print(f'   Percentage: {100 - compressed_size*100//original_size}% smaller')

print(f'\n📊 Final Request Size (with compression):')
print(f'   System prompt: ~482 tokens')
print(f'   Tools (ALL 46, compressed): ~{compressed_size // 4:,} tokens')
print(f'   User message: ~100 tokens')
print(f'   TOTAL: ~{482 + compressed_size // 4 + 100:,} tokens')
print(f'   Remaining budget: ~{12000 - (482 + compressed_size // 4 + 100):,} tokens')

print(f'\n✅ All 46 tools available!')
print('=' * 80)
