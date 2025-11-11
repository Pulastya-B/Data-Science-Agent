"""
Test Cleanup Script - Organize and remove redundant test files
Consolidates all tests into a proper structure
"""

import os
from pathlib import Path

# Current directory
root = Path(".")

# Files to KEEP (essential tests)
essential_tests = {
    # Main test suite
    "test_robust_intent.py",  # Comprehensive intent detection tests
    "quick_diagnostic.py",     # Fast validation (3 critical tests)
    
    # Official tests directory
    "tests/test_tools.py",
    "tests/test_orchestrator.py",
    
    # Examples
    "examples/titanic_example.py",
}

# Utility files to KEEP
essential_utils = {
    "test_analyzer.py",        # Test results analysis tool
    "extreme_edge_cases.py",   # Edge case documentation
    "chat_ui.py",              # UI
    "fix_string_columns.py",   # Utility
}

# Files to DELETE (redundant/outdated tests)
redundant_tests = [
    "test_tools_simple.py",
    "test_token_optimization.py",
    "test_static_check.py",
    "test_simple_code.py",
    "test_queries.py",
    "test_phase1_tools.py",
    "test_orchestrator_tools.py",
    "test_no_cache.py",
    "test_new_features.py",
    "test_intent_detection.py",           # Replaced by test_robust_intent.py
    "test_full_workflow.py",
    "test_full_pipeline.py",
    "test_fixed_multi_intent.py",         # Replaced by test_robust_intent.py
    "test_direct_code.py",
    "test_dashboard_integration.py",
    "test_custom_dashboard.py",
    "test_compression.py",
    "test_comprehensive_intent.py",       # Replaced by test_robust_intent.py
    "test_code_interpreter.py",
    "test_aggressive_compression.py",
    "run_code_test.py",
    "quick_viz_test.py",                  # Replaced by quick_diagnostic.py

]

# Utility files to DELETE (checking/debugging tools)
redundant_utils = [
    "check_registry_size.py",
    "check_prompt_size.py",
    "check_groq_models.py",
]

print("\n" + "="*80)
print("🧹 TEST CLEANUP - Removing Redundant Test Files")
print("="*80)

# Summary
total_redundant = len(redundant_tests) + len(redundant_utils)
print(f"\n📊 Summary:")
print(f"   Essential tests: {len(essential_tests)}")
print(f"   Essential utils: {len(essential_utils)}")
print(f"   Redundant files to remove: {total_redundant}")

print(f"\n✅ Files to KEEP:")
for file in sorted(essential_tests | essential_utils):
    print(f"   - {file}")

print(f"\n🗑️  Files to DELETE:")
all_redundant = redundant_tests + redundant_utils
for file in sorted(all_redundant):
    file_path = root / file
    if file_path.exists():
        print(f"   - {file} ✓ (exists)")
    else:
        print(f"   - {file} ⚠ (not found)")

print(f"\n" + "="*80)
print("⚠️  DRY RUN - No files deleted yet")
print("="*80)
print("\nTo actually delete files, uncomment the deletion code below.")
print("\nRecommended structure after cleanup:")
print("""
📁 AI Agent Data Scientist/
├── 📁 tests/                    # Official test suite
│   ├── test_tools.py
│   └── test_orchestrator.py
├── 📁 examples/
│   └── titanic_example.py
├── test_robust_intent.py        # Main test suite (12 scenarios)
├── quick_diagnostic.py          # Fast validation (3 tests)
├── test_analyzer.py             # Test results analyzer
├── extreme_edge_cases.py        # Edge case documentation
├── chat_ui.py                   # User interface
└── fix_string_columns.py        # Utility
""")

print("\n" + "="*80)
response = input("\n⚠️  Delete these files? (yes/no): ").strip().lower()

if response == 'yes':
    print("\n🗑️  Deleting redundant files...")
    deleted_count = 0
    not_found_count = 0
    
    for file in all_redundant:
        file_path = root / file
        if file_path.exists():
            try:
                os.remove(file_path)
                print(f"   ✓ Deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"   ✗ Error deleting {file}: {e}")
        else:
            not_found_count += 1
    
    print(f"\n📊 Results:")
    print(f"   ✓ Deleted: {deleted_count} files")
    print(f"   ⚠ Not found: {not_found_count} files")
    print(f"   ✅ Kept: {len(essential_tests) + len(essential_utils)} essential files")
    print(f"\n✨ Cleanup complete! Project is now organized.")
else:
    print("\n❌ Cancelled - No files deleted")

print("\n" + "="*80 + "\n")
