import re

# Read the file
with open('src/tools/advanced_training.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the string addition for object columns
object_drop_code = '''        
        # ⚠️ CRITICAL: Drop any remaining string/object columns (not encoded properly)
        object_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        object_cols = [col for col in object_cols if col != target_col]
        if object_cols:
            print(f"   ⚠️ Dropping {len(object_cols)} string columns that weren't encoded: {object_cols}")
            print(f"   💡 Categorical encoding should have been done in workflow step 8")
            df = df.drop(columns=object_cols)
'''

# Pattern to find datetime dropping code followed by 'Prepare data' comment
pattern = r'(df = df\.drop\(columns=datetime_cols\))\s*\n\s*\n\s*(# Prepare data - handle both Polars and Pandas)'

# Replacement: add object column dropping between them
replacement = r'\1' + object_drop_code + r'\n    \2'

# Apply replacement
new_content = re.sub(pattern, replacement, content)

# Count replacements
matches = len(re.findall(pattern, content))
print(f'✅ Applied object column dropping to {matches} functions')

# Write back
with open('src/tools/advanced_training.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
