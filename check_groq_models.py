"""
Script to check available Groq models and their specifications.
"""
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

print('=' * 80)
print('AVAILABLE GROQ MODELS')
print('=' * 80)

try:
    models = client.models.list()
    
    for model in models.data:
        print(f'\n📦 Model ID: {model.id}')
        print(f'   Owner: {model.owned_by}')
        print(f'   Active: {model.active}')
        
        if hasattr(model, 'context_window'):
            print(f'   Context Window: {model.context_window:,} tokens')
        
        if hasattr(model, 'type'):
            print(f'   Type: {model.type}')
            
except Exception as e:
    print(f'Error fetching models: {e}')

print('\n' + '=' * 80)
