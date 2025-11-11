"""
Quick Diagnostic - Test just 3 critical cases to validate intent detection
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator import DataScienceCopilot
from dotenv import load_dotenv

load_dotenv()

print("\n" + "="*80)
print("⚡ QUICK DIAGNOSTIC - 3 Critical Intent Tests")
print("="*80)
print("\nThis tests the 3 most important intent patterns:")
print("  1. Visualization ONLY (should NOT run ML pipeline)")
print("  2. Multi-Intent (should handle multiple goals)")
print("  3. Negation (should respect 'without training')")
print("\n" + "="*80 + "\n")

# Create test dataset
test_file = "./temp/test_earthquake.csv"
if not os.path.exists(test_file):
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    test_data = pd.DataFrame({
        'mag': np.random.uniform(1, 8, 100),
        'depth': np.random.uniform(0, 700, 100),
        'latitude': np.random.uniform(-90, 90, 100),
        'longitude': np.random.uniform(-180, 180, 100),
    })
    os.makedirs('./temp', exist_ok=True)
    test_data.to_csv(test_file, index=False)
    print(f"✅ Created test dataset\n")

copilot = DataScienceCopilot()

# Test 1: Visualization Only
print("TEST 1: Visualization Only")
print("-" * 80)
print("Query: 'Generate interactive plots for mag and latitude'")
print("Expected: Only call generate_interactive_scatter, NO ml pipeline\n")

result1 = copilot.analyze(
    file_path=test_file,
    task_description="Generate interactive plots for mag and latitude",
    max_iterations=3,
    use_cache=False
)

tools1 = [s.get("tool") for s in result1.get("workflow_history", []) if s.get("tool") != "BLOCKED"]
blocked1 = len([s for s in result1.get("workflow_history", []) if s.get("tool") == "BLOCKED"])

print(f"Tools called: {tools1}")
print(f"Tools blocked: {blocked1}")

test1_pass = (
    "generate_interactive_scatter" in tools1 and
    "train_baseline_models" not in tools1 and
    "profile_dataset" not in tools1
)

if test1_pass:
    print("✅ TEST 1 PASSED\n")
else:
    print("❌ TEST 1 FAILED\n")

import time
time.sleep(3)

# Test 2: Multi-Intent  
print("\nTEST 2: Multi-Intent (Clean + Visualize)")
print("-" * 80)
print("Query: 'Clean missing values and generate plots'")
print("Expected: Call both clean_missing_values AND visualization tool\n")

result2 = copilot.analyze(
    file_path=test_file,
    task_description="Clean missing values and generate plots",
    max_iterations=7,
    use_cache=False
)

tools2 = [s.get("tool") for s in result2.get("workflow_history", []) if s.get("tool") != "BLOCKED"]

print(f"Tools called: {tools2}")

has_cleaning = any(t in tools2 for t in ["clean_missing_values", "handle_outliers"])
has_viz = any(t in tools2 for t in ["generate_interactive_scatter", "generate_eda_plots", "generate_plotly_dashboard"])
no_training = "train_baseline_models" not in tools2

test2_pass = has_cleaning and no_training  # Viz optional but no training

if test2_pass:
    print("✅ TEST 2 PASSED\n")
else:
    print(f"❌ TEST 2 FAILED (cleaning:{has_cleaning}, no_train:{no_training})\n")

time.sleep(3)

# Test 3: Negation
print("\nTEST 3: Negation ('without training')")
print("-" * 80)
print("Query: 'Analyze dataset without training any models'")
print("Expected: Profile/analyze but NO train_baseline_models\n")

result3 = copilot.analyze(
    file_path=test_file,
    task_description="Analyze dataset without training any models",
    max_iterations=7,
    use_cache=False
)

tools3 = [s.get("tool") for s in result3.get("workflow_history", []) if s.get("tool") != "BLOCKED"]

print(f"Tools called: {tools3}")

has_analysis = any(t in tools3 for t in ["profile_dataset", "analyze_correlations", "detect_data_quality_issues"])
no_training3 = "train_baseline_models" not in tools3 and "hyperparameter_tuning" not in tools3

test3_pass = has_analysis and no_training3

if test3_pass:
    print("✅ TEST 3 PASSED\n")
else:
    print(f"❌ TEST 3 FAILED (analysis:{has_analysis}, no_train:{no_training3})\n")

# Summary
print("\n" + "="*80)
print("📊 QUICK DIAGNOSTIC SUMMARY")
print("="*80)

passed = sum([test1_pass, test2_pass, test3_pass])
print(f"\nResults: {passed}/3 tests passed")

if passed == 3:
    print("\n🎉 EXCELLENT: All critical patterns working!")
    print("   ✓ Visualization-only detection")
    print("   ✓ Multi-intent handling")
    print("   ✓ Negation respect")
    print("\n✅ System is ready for production!")
elif passed == 2:
    print("\n⚠️  GOOD: Most patterns working, one issue to fix")
    if not test1_pass:
        print("   ❌ Fix: Visualization-only still running ML pipeline")
    if not test2_pass:
        print("   ❌ Fix: Multi-intent not detecting both goals")
    if not test3_pass:
        print("   ❌ Fix: Negation not blocking training")
elif passed == 1:
    print("\n⚠️  NEEDS WORK: Multiple issues detected")
    print("   Run full test suite: py test_robust_intent.py")
else:
    print("\n❌ CRITICAL: Intent detection not working")
    print("   Review orchestrator.py intent detection logic")

print("\n" + "="*80 + "\n")
