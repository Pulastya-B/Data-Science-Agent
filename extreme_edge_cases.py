"""
EXTREME EDGE CASE TESTS - Really challenging queries
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator import DataScienceCopilot
from dotenv import load_dotenv

load_dotenv()

# Extreme edge cases that LLMs typically struggle with
extreme_test_cases = [
    {
        "name": "Contradictory Request",
        "query": "Train a model but don't use any machine learning",
        "expected_behavior": "Should ask for clarification OR ignore 'train' due to negation",
        "must_NOT_call": ["train_baseline_models"]
    },
    {
        "name": "Circular Logic",
        "query": "Clean the data before you profile it, but only clean what profiling shows needs cleaning",
        "expected_behavior": "Should profile first, then clean (break the circle)",
        "must_call": ["profile_dataset", "clean_missing_values"]
    },
    {
        "name": "Overly Specific Multi-Step",
        "query": "First drop missing values in column mag, then create year/month/day features from date, then encode location with one-hot encoding, then train XGBoost with max_depth=10",
        "expected_behavior": "Should execute all steps in order",
        "must_call": ["clean_missing_values", "create_time_features", "encode_categorical", "train_baseline_models"]
    },
    {
        "name": "Implicit AND Explicit",
        "query": "Visualize the magnitude distribution and also create a dashboard",
        "expected_behavior": "Should call both histogram and dashboard tools",
        "must_call": ["generate_interactive_histogram"],  # Or similar viz tool
        "can_call": ["generate_plotly_dashboard"]
    },
    {
        "name": "Negative Framing",
        "query": "I don't want to train models, I just want to see the data",
        "expected_behavior": "Should only do profiling/visualization",
        "must_call": ["profile_dataset"],
        "must_NOT_call": ["train_baseline_models"]
    },
    {
        "name": "Question Format",
        "query": "What are the missing values in this dataset?",
        "expected_behavior": "Should detect as profiling/analysis request",
        "must_call": ["detect_data_quality_issues"],
        "must_NOT_call": ["train_baseline_models"]
    },
    {
        "name": "Comparison Request",
        "query": "Which is better for predicting magnitude: depth or latitude?",
        "expected_behavior": "Should train model and analyze feature importance",
        "must_call": ["train_baseline_models"]
    },
    {
        "name": "Conditional Request",
        "query": "If there are missing values, clean them, otherwise just visualize",
        "expected_behavior": "Should check for missing values, then clean if found",
        "must_call": ["detect_data_quality_issues"],
        "can_call": ["clean_missing_values"]  # Optional depending on data
    },
    {
        "name": "Domain Jargon",
        "query": "Perform hyperparameter optimization for the best regressor",
        "expected_behavior": "Should understand this means ML training + tuning",
        "must_call": ["train_baseline_models", "hyperparameter_tuning"]
    },
    {
        "name": "Vague Request",
        "query": "Make this dataset better",
        "expected_behavior": "Should default to cleaning + feature engineering",
        "can_call": ["clean_missing_values", "handle_outliers", "create_time_features"]
    },
    {
        "name": "Sequential with 'Then'",
        "query": "Profile the dataset, then clean it, then visualize the results",
        "expected_behavior": "Should execute in exact order",
        "must_call": ["profile_dataset", "clean_missing_values"],
        "can_call": ["generate_eda_plots", "generate_plotly_dashboard"]
    },
    {
        "name": "Mixed Natural Language",
        "query": "Can you help me understand the earthquake patterns by looking at magnitude over time and also checking if location affects depth?",
        "expected_behavior": "Exploratory analysis with time series and correlation",
        "must_call": ["profile_dataset"],
        "can_call": ["analyze_correlations", "generate_interactive_time_series"],
        "must_NOT_call": ["train_baseline_models"]  # No explicit training request
    }
]

print("\n" + "="*100)
print("💀 EXTREME EDGE CASE TESTS - LLM Torture Test")
print("="*100)
print("\nThese tests are designed to break intent detection systems.")
print("Expected behavior: At least 70% should handle gracefully\n")

print(f"Total Extreme Cases: {len(extreme_test_cases)}\n")
print("Test Categories:")
print("  - Contradictory requests (logic puzzles)")
print("  - Circular dependencies")
print("  - Overly specific instructions")
print("  - Negative framing")
print("  - Question formats")
print("  - Domain jargon")
print("  - Vague/ambiguous requests")
print("  - Sequential multi-step")
print("  - Natural language complexity")

print("\n" + "="*100)
print("📋 TEST PREVIEW (Run manually with test_robust_intent.py)")
print("="*100)

for i, test in enumerate(extreme_test_cases, 1):
    print(f"\n{i}. {test['name']}")
    print(f"   Query: \"{test['query']}\"")
    print(f"   Expected: {test['expected_behavior']}")
    if test.get('must_call'):
        print(f"   Must call: {test['must_call']}")
    if test.get('must_NOT_call'):
        print(f"   Must NOT call: {test['must_NOT_call']}")

print("\n" + "="*100)
print("💡 HOW TO RUN THESE TESTS:")
print("="*100)
print("""
1. Add these test cases to test_robust_intent.py
2. Update test_subset to include extreme cases
3. Run: py test_robust_intent.py
4. Analyze which edge cases fail
5. Improve keyword detection or add special handling

Example:
```python
# In test_robust_intent.py, add:
test_cases.extend(extreme_test_cases_from_extreme_tests)
test_subset = test_cases[0:15]  # Include extreme tests
```
""")

print("\n" + "="*100)
print("🎯 EXPECTED CHALLENGES:")
print("="*100)
print("""
Likely to PASS:
  ✅ Negative framing ("I don't want to train")
  ✅ Question format ("What are the missing values?")
  ✅ Domain jargon ("hyperparameter optimization")
  ✅ Sequential with 'then'

Likely to STRUGGLE:
  ⚠️  Contradictory requests (need conflict resolution)
  ⚠️  Circular logic (need dependency analysis)
  ⚠️  Conditional requests ("if...then...")
  ⚠️  Vague requests ("make better" - too ambiguous)

Will LEARN FROM FAILURES:
  📚 Which keywords to add
  📚 Which patterns need special handling
  📚 When to ask user for clarification
""")

print("\n" + "="*100 + "\n")
