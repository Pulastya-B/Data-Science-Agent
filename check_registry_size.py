"""
Script to check tools registry size and optimize it.
"""
import json
import sys
sys.path.append('src')

from tools.tools_registry import TOOLS

# Calculate size
tools_json = json.dumps(TOOLS, indent=2)
tools_compact = json.dumps(TOOLS)

print('=' * 80)
print('TOOLS REGISTRY SIZE ANALYSIS')
print('=' * 80)
print(f'\nNumber of tools: {len(TOOLS)}')
print(f'JSON size (pretty): {len(tools_json):,} characters')
print(f'JSON size (compact): {len(tools_compact):,} characters')
print(f'Estimated tokens (pretty): {len(tools_json) // 4:,}')
print(f'Estimated tokens (compact): {len(tools_compact) // 4:,}')

print('\n' + '=' * 80)
print('SAMPLE TOOL DEFINITION:')
print('=' * 80)
print(json.dumps(TOOLS[0], indent=2)[:500])
print('\n...')

print('\n' + '=' * 80)
print('TOTAL REQUEST SIZE ESTIMATE:')
print('=' * 80)
print(f'System prompt: ~482 tokens')
print(f'Tools registry: ~{len(tools_compact) // 4:,} tokens')
print(f'User message: ~100 tokens')
print(f'Tool results (per iteration): ~500-2000 tokens')
print(f'TOTAL (first call): ~{482 + len(tools_compact) // 4 + 100:,} tokens')
print(f'TOTAL (with 5 tools): ~{482 + len(tools_compact) // 4 + 100 + 2000:,} tokens')
