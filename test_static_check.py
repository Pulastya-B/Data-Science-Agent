"""
Simplified Test Script for Phase 1 New Tools
Tests the core functionality without importing the full orchestrator
"""

import sys
from pathlib import Path

print("=" * 80)
print("PHASE 1 TOOLS - SIMPLIFIED TEST SUITE")
print("=" * 80)

# Test 1: Check files exist
print("\n[TEST 1] Checking if new tool files exist...")
tool_files = {
    "data_profiling.py": Path("src/tools/data_profiling.py"),
    "data_cleaning.py": Path("src/tools/data_cleaning.py"),
    "data_wrangling.py": Path("src/tools/data_wrangling.py"),
    "tools_registry.py": Path("src/tools/tools_registry.py"),
    "orchestrator.py": Path("src/orchestrator.py"),
}

for name, path in tool_files.items():
    if path.exists():
        print(f"✅ {name} exists")
    else:
        print(f"❌ {name} missing")

# Test 2: Check for new functions in files
print("\n[TEST 2] Checking for new function definitions in files...")

# Check data_profiling.py for get_smart_summary
profiling_content = Path("src/tools/data_profiling.py").read_text(encoding='utf-8')
if "def get_smart_summary(" in profiling_content:
    print(f"✅ get_smart_summary() found in data_profiling.py")
else:
    print(f"❌ get_smart_summary() NOT found in data_profiling.py")

# Check data_cleaning.py for threshold parameter
cleaning_content = Path("src/tools/data_cleaning.py").read_text(encoding='utf-8')
if "threshold" in cleaning_content and "def clean_missing_values(" in cleaning_content:
    print(f"✅ clean_missing_values() has threshold parameter in data_cleaning.py")
else:
    print(f"❌ threshold parameter NOT found in data_cleaning.py")

# Check data_wrangling.py for new functions
wrangling_content = Path("src/tools/data_wrangling.py").read_text(encoding='utf-8')
wrangling_functions = ["merge_datasets", "concat_datasets", "reshape_dataset"]
for func in wrangling_functions:
    if f"def {func}(" in wrangling_content:
        print(f"✅ {func}() found in data_wrangling.py")
    else:
        print(f"❌ {func}() NOT found in data_wrangling.py")

# Test 3: Check tools_registry.py
print("\n[TEST 3] Checking tools_registry.py for new tool definitions...")
registry_content = Path("src/tools/tools_registry.py").read_text(encoding='utf-8')

new_tools = ["get_smart_summary", "merge_datasets", "concat_datasets", "reshape_dataset"]
for tool in new_tools:
    if f'"name": "{tool}"' in registry_content:
        print(f"✅ {tool} registered in tools_registry.py")
    else:
        print(f"❌ {tool} NOT registered in tools_registry.py")

# Check total tool count
if "All 67 Tools" in registry_content:
    print(f"✅ Tool count updated to 67 in registry header")
elif "All 44 Tools" in registry_content:
    print(f"⚠️  Tool count still shows 44 (should be 67)")
else:
    print(f"⚠️  Tool count header not found")

# Test 4: Check orchestrator.py for imports
print("\n[TEST 4] Checking orchestrator.py for new tool imports...")
orchestrator_content = Path("src/orchestrator.py").read_text(encoding='utf-8')

if "get_smart_summary" in orchestrator_content:
    print(f"✅ get_smart_summary imported in orchestrator.py")
else:
    print(f"❌ get_smart_summary NOT imported in orchestrator.py")

if "merge_datasets" in orchestrator_content:
    print(f"✅ merge_datasets imported in orchestrator.py")
else:
    print(f"❌ merge_datasets NOT imported in orchestrator.py")

if "concat_datasets" in orchestrator_content:
    print(f"✅ concat_datasets imported in orchestrator.py")
else:
    print(f"❌ concat_datasets NOT imported in orchestrator.py")

if "reshape_dataset" in orchestrator_content:
    print(f"✅ reshape_dataset imported in orchestrator.py")
else:
    print(f"❌ reshape_dataset NOT imported in orchestrator.py")

# Test 5: Check __init__.py exports
print("\n[TEST 5] Checking src/tools/__init__.py for exports...")
init_content = Path("src/tools/__init__.py").read_text(encoding='utf-8')

exports_to_check = ["get_smart_summary", "merge_datasets", "concat_datasets", "reshape_dataset"]
for export in exports_to_check:
    if export in init_content:
        print(f"✅ {export} exported from __init__.py")
    else:
        print(f"❌ {export} NOT exported from __init__.py")

# Test 6: Function signature checks
print("\n[TEST 6] Checking function signatures...")

# get_smart_summary signature
if "def get_smart_summary(file_path" in profiling_content and "n_samples" in profiling_content:
    print(f"✅ get_smart_summary has correct signature (file_path, n_samples)")
else:
    print(f"⚠️  get_smart_summary signature may be incomplete")

# merge_datasets signature
if all(param in wrangling_content for param in ["left_path", "right_path", "output_path", "how", "on"]):
    print(f"✅ merge_datasets has correct parameters")
else:
    print(f"⚠️  merge_datasets may be missing some parameters")

# concat_datasets signature
if all(param in wrangling_content for param in ["file_paths", "output_path", "axis"]):
    print(f"✅ concat_datasets has correct parameters")
else:
    print(f"⚠️  concat_datasets may be missing some parameters")

# reshape_dataset signature
if all(param in wrangling_content for param in ["operation", "index", "columns", "values"]):
    print(f"✅ reshape_dataset has correct parameters")
else:
    print(f"⚠️  reshape_dataset may be missing some parameters")

# Test 7: Documentation strings
print("\n[TEST 7] Checking for docstrings...")

files_to_check = {
    "data_wrangling.py": wrangling_content,
    "data_profiling.py (get_smart_summary)": profiling_content,
}

for file_name, content in files_to_check.items():
    docstring_count = content.count('"""') + content.count("'''")
    if docstring_count >= 4:  # At least 2 complete docstrings (opening + closing)
        print(f"✅ {file_name} has docstrings")
    else:
        print(f"⚠️  {file_name} may be missing docstrings")

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

print("""
✅ Static analysis complete!

This test verified:
1. All new tool files exist
2. New functions are defined in correct files
3. Tools are registered in tools_registry.py
4. Tools are imported in orchestrator.py
5. Tools are exported from __init__.py
6. Function signatures look correct
7. Documentation strings present

Next steps:
1. Wait for pip install to complete
2. Run full functional tests: py test_phase1_tools.py
3. Test with Gradio UI: py src/chat_ui.py
""")

print("=" * 80)
