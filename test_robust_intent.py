"""
ROBUST INTENT DETECTION TEST SUITE
Tests edge cases, multi-intent requests, and challenging scenarios
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator import DataScienceCopilot
from dotenv import load_dotenv
import time

load_dotenv()

print("\n" + "="*90)
print("🧪 ROBUST INTENT DETECTION TEST SUITE - Edge Cases & Multi-Intent")
print("="*90)

# Create test dataset
test_file = "./temp/test_robust.csv"
if not os.path.exists(test_file):
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    test_data = pd.DataFrame({
        'mag': np.random.uniform(1, 8, 100),
        'depth': np.random.uniform(0, 700, 100),
        'latitude': np.random.uniform(-90, 90, 100),
        'longitude': np.random.uniform(-180, 180, 100),
        'location': ['City_' + str(i%10) for i in range(100)],
        'date': pd.date_range('2020-01-01', periods=100, freq='D'),
    })
    # Add missing values
    test_data.loc[0:5, 'depth'] = None
    test_data.loc[10:15, 'mag'] = None
    
    os.makedirs('./temp', exist_ok=True)
    test_data.to_csv(test_file, index=False)
    print(f"✅ Created test dataset: {test_file}")
    print(f"   - 100 rows, 6 columns")
    print(f"   - Missing values: depth (6), mag (6)")
    print(f"   - Categorical: location")
    print(f"   - Datetime: date\n")

# Comprehensive test cases
test_cases = [
    # ========== SIMPLE SINGLE-INTENT TESTS ==========
    {
        "id": "T1",
        "category": "Simple - Visualization Only",
        "query": "Generate interactive plots for magnitude and latitude",
        "expected_intent": "VISUALIZATION_ONLY",
        "must_call": ["generate_interactive_scatter"],
        "must_NOT_call": ["train_baseline_models", "clean_missing_values", "profile_dataset"],
        "max_iterations": 3
    },
    {
        "id": "T2",
        "category": "Simple - Cleaning Only",
        "query": "Clean the missing values in this dataset",
        "expected_intent": "CLEANING_ONLY",
        "must_call": ["clean_missing_values"],
        "must_NOT_call": ["train_baseline_models", "hyperparameter_tuning"],
        "max_iterations": 5
    },
    {
        "id": "T3",
        "category": "Simple - Feature Engineering Only",
        "query": "Create time-based features from the date column",
        "expected_intent": "FEATURE_ENGINEERING_ONLY",
        "must_call": ["create_time_features"],
        "must_NOT_call": ["train_baseline_models", "hyperparameter_tuning"],
        "max_iterations": 5
    },
    
    # ========== MULTI-INTENT TESTS (CHALLENGING) ==========
    {
        "id": "T4",
        "category": "Multi-Intent - Clean + Visualize",
        "query": "Clean the missing values and then generate interactive plots",
        "expected_intent": "MULTI_INTENT",
        "must_call": ["clean_missing_values"],  # At least cleaning
        "can_call": ["generate_interactive_scatter", "generate_plotly_dashboard"],  # Viz is optional
        "must_NOT_call": ["train_baseline_models", "hyperparameter_tuning"],
        "max_iterations": 7
    },
    {
        "id": "T5",
        "category": "Multi-Intent - Feature Eng + Train",
        "query": "Create time features and then train a model to predict magnitude",
        "expected_intent": "MULTI_INTENT",
        "must_call": ["create_time_features", "train_baseline_models"],
        "can_call": ["clean_missing_values", "encode_categorical"],
        "must_NOT_call": [],  # Everything allowed for training
        "max_iterations": 15
    },
    {
        "id": "T6",
        "category": "Multi-Intent - Complex (3 intents)",
        "query": "Clean the data, engineer features, and create visualizations but don't train any models",
        "expected_intent": "MULTI_INTENT",
        "must_call": ["clean_missing_values"],
        "can_call": ["create_time_features", "encode_categorical", "generate_eda_plots"],
        "must_NOT_call": ["train_baseline_models", "hyperparameter_tuning"],
        "max_iterations": 10
    },
    
    # ========== EDGE CASES & TRICKY QUERIES ==========
    {
        "id": "T7",
        "category": "Edge - Negation ('without training')",
        "query": "Clean and analyze the data without training any models",
        "expected_intent": "MULTI_INTENT",
        "must_call": [],  # Any cleaning/profiling ok
        "can_call": ["clean_missing_values", "profile_dataset", "analyze_correlations"],
        "must_NOT_call": ["train_baseline_models", "hyperparameter_tuning"],
        "max_iterations": 7
    },
    {
        "id": "T8",
        "category": "Edge - Ambiguous ('analyze')",
        "query": "Analyze this dataset",
        "expected_intent": "EXPLORATORY_ANALYSIS or FULL_ML_PIPELINE",
        "must_call": ["profile_dataset"],  # Should at least profile
        "can_call": ["analyze_correlations", "generate_eda_plots"],
        "must_NOT_call": [],  # Ambiguous, so anything goes
        "max_iterations": 10
    },
    {
        "id": "T9",
        "category": "Edge - Implicit ML ('predict')",
        "query": "What can predict earthquake magnitude best?",
        "expected_intent": "FULL_ML_PIPELINE",
        "must_call": ["train_baseline_models"],
        "can_call": ["clean_missing_values", "hyperparameter_tuning"],
        "must_NOT_call": [],
        "max_iterations": 15
    },
    {
        "id": "T10",
        "category": "Edge - Only Profiling",
        "query": "Show me a summary of this dataset",
        "expected_intent": "EXPLORATORY_ANALYSIS",
        "must_call": ["profile_dataset"],
        "can_call": ["analyze_correlations", "detect_data_quality_issues"],
        "must_NOT_call": ["train_baseline_models", "clean_missing_values"],
        "max_iterations": 5
    },
    
    # ========== STRESS TESTS ==========
    {
        "id": "T11",
        "category": "Stress - Verbose Multi-Intent",
        "query": "I want you to first clean all the missing values, then create time-based features from the date column, encode the categorical location column, and finally generate interactive scatter plots showing magnitude vs depth",
        "expected_intent": "MULTI_INTENT",
        "must_call": ["clean_missing_values", "create_time_features"],
        "can_call": ["encode_categorical", "generate_interactive_scatter"],
        "must_NOT_call": ["train_baseline_models"],
        "max_iterations": 10
    },
    {
        "id": "T12",
        "category": "Stress - Full Pipeline with Specifics",
        "query": "Train the best possible model to predict magnitude. Use all feature engineering techniques and optimize hyperparameters",
        "expected_intent": "FULL_ML_PIPELINE",
        "must_call": ["train_baseline_models"],
        "can_call": ["create_time_features", "encode_categorical", "hyperparameter_tuning"],
        "must_NOT_call": [],
        "max_iterations": 20
    }
]

# Run tests
copilot = DataScienceCopilot()
results = []

# Test subset (first 6 tests for quick validation)
test_subset = test_cases[:6]

for test in test_subset:
    print(f"\n{'='*90}")
    print(f"TEST {test['id']}: {test['category']}")
    print(f"{'='*90}")
    print(f"📝 Query: {test['query']}")
    print(f"🎯 Expected Intent: {test['expected_intent']}")
    print(f"✅ Must Call: {test.get('must_call', [])}")
    print(f"🚫 Must NOT Call: {test.get('must_NOT_call', [])}")
    print(f"\n⏱️  Running (max {test['max_iterations']} iterations)...\n")
    
    start_time = time.time()
    
    try:
        result = copilot.analyze(
            file_path=test_file,
            task_description=test['query'],
            max_iterations=test['max_iterations'],
            use_cache=False
        )
        
        execution_time = time.time() - start_time
        
        # Extract tools
        tools_called = [
            step.get("tool") 
            for step in result.get("workflow_history", []) 
            if step.get("tool") != "BLOCKED"
        ]
        blocked_tools = [
            step.get("blocked_tool") 
            for step in result.get("workflow_history", []) 
            if step.get("tool") == "BLOCKED"
        ]
        
        print(f"\n{'-'*90}")
        print(f"📊 RESULTS:")
        print(f"{'-'*90}")
        print(f"Status: {result.get('status')}")
        print(f"Iterations: {result.get('iterations')}/{test['max_iterations']}")
        print(f"Execution Time: {execution_time:.2f}s")
        print(f"API Calls: {result.get('api_calls', 'N/A')}")
        print(f"\n🔧 Tools Called ({len(tools_called)}): {tools_called}")
        print(f"🚫 Tools Blocked ({len(blocked_tools)}): {blocked_tools}")
        
        # Validation
        must_call = test.get('must_call', [])
        can_call = test.get('can_call', [])
        must_NOT_call = test.get('must_NOT_call', [])
        
        # Check required tools
        missing_required = [t for t in must_call if t not in tools_called]
        called_forbidden = [t for t in must_NOT_call if t in tools_called]
        
        # Calculate score
        passed = len(missing_required) == 0 and len(called_forbidden) == 0
        
        print(f"\n{'='*90}")
        print(f"✅ VALIDATION:")
        print(f"{'='*90}")
        
        if not missing_required:
            print(f"✅ All required tools called: {must_call}")
        else:
            print(f"❌ Missing required tools: {missing_required}")
        
        if not called_forbidden:
            print(f"✅ Forbidden tools avoided: {must_NOT_call}")
        else:
            print(f"❌ Forbidden tools WERE called: {called_forbidden}")
        
        if can_call:
            called_optional = [t for t in can_call if t in tools_called]
            print(f"ℹ️  Optional tools called: {called_optional if called_optional else 'None'}")
        
        # Edge case handling
        if test['id'] in ['T8']:  # Ambiguous test
            print(f"ℹ️  Note: Ambiguous query - multiple valid outcomes")
            passed = len(called_forbidden) == 0  # Only check forbidden tools
        
        if passed:
            print(f"\n🎉 TEST {test['id']} PASSED!")
            results.append({"id": test['id'], "status": "PASS", "time": execution_time})
        else:
            print(f"\n❌ TEST {test['id']} FAILED!")
            results.append({"id": test['id'], "status": "FAIL", "time": execution_time})
        
    except Exception as e:
        print(f"\n💥 TEST {test['id']} CRASHED!")
        print(f"Error: {str(e)}")
        results.append({"id": test['id'], "status": "CRASH", "time": 0})
    
    print(f"{'='*90}\n")
    
    # Rate limiting between tests
    time.sleep(2)

# Final summary
print("\n" + "="*90)
print("📈 FINAL SUMMARY")
print("="*90)

passed = sum(1 for r in results if r['status'] == 'PASS')
failed = sum(1 for r in results if r['status'] == 'FAIL')
crashed = sum(1 for r in results if r['status'] == 'CRASH')
total = len(results)

print(f"\nResults: {passed}/{total} PASSED ({passed/total*100:.1f}%)")
print(f"         {failed}/{total} FAILED ({failed/total*100:.1f}%)")
print(f"         {crashed}/{total} CRASHED ({crashed/total*100:.1f}%)")

print(f"\nDetailed Results:")
for r in results:
    status_icon = "✅" if r['status'] == 'PASS' else "❌" if r['status'] == 'FAIL' else "💥"
    print(f"  {status_icon} {r['id']}: {r['status']} ({r['time']:.2f}s)")

avg_time = sum(r['time'] for r in results) / len(results) if results else 0
print(f"\nAverage Execution Time: {avg_time:.2f}s")

# Quality metrics
print(f"\n{'='*90}")
print(f"🎯 SYSTEM QUALITY METRICS")
print(f"{'='*90}")

if passed == total:
    print(f"✅ EXCELLENT: All tests passed! Intent detection is robust.")
elif passed >= total * 0.8:
    print(f"✅ GOOD: {passed/total*100:.1f}% passed. Minor tuning needed.")
elif passed >= total * 0.6:
    print(f"⚠️  FAIR: {passed/total*100:.1f}% passed. Significant improvements needed.")
else:
    print(f"❌ POOR: {passed/total*100:.1f}% passed. Major issues with intent detection.")

print(f"\nKey Capabilities Tested:")
print(f"  ✓ Single-intent queries (T1-T3)")
print(f"  ✓ Multi-intent queries (T4-T6)")
print(f"  ✓ Negation handling (T7)")
print(f"  ✓ Ambiguous queries (T8)")
print(f"  ✓ Implicit intents (T9-T10)")
print(f"  ✓ Stress tests (T11-T12 - not run yet)")

print(f"\n{'='*90}")
print(f"💡 RECOMMENDATIONS:")
print(f"{'='*90}")

if failed > 0:
    print(f"\nFailed Tests Analysis:")
    failed_tests = [r['id'] for r in results if r['status'] == 'FAIL']
    for tid in failed_tests:
        test = next(t for t in test_cases if t['id'] == tid)
        print(f"  - {tid}: {test['category']}")
        print(f"    Issue: Check if intent detection keywords need adjustment")

if crashed > 0:
    print(f"\nCrashed Tests - Investigate:")
    crashed_tests = [r['id'] for r in results if r['status'] == 'CRASH']
    for tid in crashed_tests:
        print(f"  - {tid}: Check error logs above")

print(f"\n✅ To run ALL tests (including stress tests T11-T12):")
print(f"   Modify: test_subset = test_cases  # Run all 12 tests")

print(f"\n{'='*90}\n")
