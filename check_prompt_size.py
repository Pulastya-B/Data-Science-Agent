"""
Script to estimate token count for system prompt.
Rough estimate: 1 token ≈ 4 characters for English text.
"""

def estimate_tokens(text):
    """Rough token estimation."""
    return len(text) // 4

# Read orchestrator to get system prompt
import sys
sys.path.append('src')

from orchestrator import DataScienceCopilot
import os
from dotenv import load_dotenv

load_dotenv()

# Create instance to get system prompt
copilot = DataScienceCopilot(groq_api_key=os.getenv('GROQ_API_KEY'))
system_prompt = copilot._build_system_prompt()

print('=' * 80)
print('SYSTEM PROMPT TOKEN ESTIMATION')
print('=' * 80)
print(f'\nCharacters: {len(system_prompt):,}')
print(f'Estimated Tokens: {estimate_tokens(system_prompt):,}')
print(f'\nModel: {os.getenv("GROQ_MODEL")}')
print(f'Rate Limit: 12,000 TPM')
print(f'\n✅ Prompt fits within limit!' if estimate_tokens(system_prompt) < 10000 else '⚠️ Prompt may be too large!')
print('\n' + '=' * 80)
print('\nPROMPT PREVIEW (first 500 chars):')
print('-' * 80)
print(system_prompt[:500])
print('\n...')
