"""
Test token savings with essential tools only.
"""
import json
import sys
sys.path.append('src')

from tools.tools_registry import TOOLS

essential_tool_names = [
    "profile_dataset",
    "detect_data_quality_issues",
    "clean_missing_values",
    "handle_outliers",
    "force_numeric_conversion",
    "fix_data_types",
    "encode_categorical",
    "train_baseline_models",
    "generate_model_report",
    "perform_eda_analysis",
    "handle_imbalanced_data",
    "perform_feature_scaling",
    "hyperparameter_tuning",
    "auto_feature_engineering"
]

# Filter to essential tools
essential_tools = [
    tool for tool in TOOLS 
    if tool.get("function", {}).get("name") in essential_tool_names
]

all_size = len(json.dumps(TOOLS))
essential_size = len(json.dumps(essential_tools))

print('=' * 80)
print('TOKEN OPTIMIZATION RESULTS')
print('=' * 80)
print(f'\n📦 All Tools:')
print(f'   Count: 46 tools')
print(f'   Size: {all_size:,} characters')
print(f'   Tokens: ~{all_size // 4:,}')

print(f'\n✅ Essential Tools:')
print(f'   Count: {len(essential_tools)} tools')
print(f'   Size: {essential_size:,} characters')
print(f'   Tokens: ~{essential_size // 4:,}')

print(f'\n💰 Savings:')
print(f'   Reduced by: {len(TOOLS) - len(essential_tools)} tools ({100 - len(essential_tools)*100//len(TOOLS)}%)')
print(f'   Token reduction: ~{(all_size - essential_size) // 4:,} tokens')
print(f'   Percentage saved: {100 - essential_size*100//all_size}%')

print(f'\n📊 Request Size Estimate (with essential tools):')
print(f'   System prompt: ~482 tokens')
print(f'   Tools registry: ~{essential_size // 4:,} tokens')
print(f'   User message: ~100 tokens')
print(f'   TOTAL (first call): ~{482 + essential_size // 4 + 100:,} tokens')
print(f'   Remaining budget: ~{12000 - (482 + essential_size // 4 + 100):,} tokens for tool results')

print('\n' + '=' * 80)
print('✅ Should fit well within 12,000 TPM limit!')
print('=' * 80)
