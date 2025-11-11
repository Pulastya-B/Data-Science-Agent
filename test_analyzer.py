"""
Test Results Analyzer - Identifies patterns in test failures
"""

def analyze_test_results(results, test_cases):
    """
    Analyze test results to identify patterns and provide actionable insights.
    
    Args:
        results: List of test results with {id, status, time, tools_called, etc.}
        test_cases: Original test case definitions
    
    Returns:
        Analysis report with patterns and recommendations
    """
    
    print("\n" + "="*100)
    print("🔍 DETAILED TEST ANALYSIS")
    print("="*100)
    
    # Categorize failures
    failed_by_category = {}
    failed_by_intent = {}
    
    for result in results:
        if result['status'] == 'FAIL':
            test = next(t for t in test_cases if t['id'] == result['id'])
            category = test['category'].split(' - ')[0]  # e.g., "Simple", "Multi-Intent"
            intent = test.get('expected_intent', 'Unknown')
            
            failed_by_category[category] = failed_by_category.get(category, 0) + 1
            failed_by_intent[intent] = failed_by_intent.get(intent, 0) + 1
    
    # Failure patterns
    print("\n📊 FAILURE PATTERNS:")
    print("-" * 100)
    
    if failed_by_category:
        print("\nBy Category:")
        for category, count in sorted(failed_by_category.items(), key=lambda x: x[1], reverse=True):
            print(f"  ❌ {category}: {count} failures")
    else:
        print("\n  ✅ No failures by category!")
    
    if failed_by_intent:
        print("\nBy Intent Type:")
        for intent, count in sorted(failed_by_intent.items(), key=lambda x: x[1], reverse=True):
            print(f"  ❌ {intent}: {count} failures")
    else:
        print("\n  ✅ No failures by intent!")
    
    # Performance analysis
    print("\n⚡ PERFORMANCE ANALYSIS:")
    print("-" * 100)
    
    if results:
        times = [r['time'] for r in results if r['time'] > 0]
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\nExecution Time:")
            print(f"  Average: {avg_time:.2f}s")
            print(f"  Fastest: {min_time:.2f}s")
            print(f"  Slowest: {max_time:.2f}s")
            
            # Identify slow tests
            slow_threshold = avg_time * 1.5
            slow_tests = [r for r in results if r['time'] > slow_threshold]
            if slow_tests:
                print(f"\n  ⚠️  Slow tests (>{slow_threshold:.2f}s):")
                for r in slow_tests:
                    test = next(t for t in test_cases if t['id'] == r['id'])
                    print(f"    - {r['id']}: {r['time']:.2f}s ({test['category']})")
    
    # Common failure reasons
    print("\n🔧 RECOMMENDATIONS:")
    print("-" * 100)
    
    recommendations = []
    
    # Check for visualization failures
    viz_failures = [r for r in results if r['status'] == 'FAIL']
    for result in viz_failures:
        test = next(t for t in test_cases if t['id'] == result['id'])
        if 'VISUALIZATION' in test.get('expected_intent', ''):
            recommendations.append(
                "📌 Visualization tests failing: Check if wrong viz tools being called "
                "(e.g., generate_data_quality_plots vs generate_interactive_scatter)"
            )
            break
    
    # Check for multi-intent failures
    multi_intent_failures = [r for r in results if r['status'] == 'FAIL']
    for result in multi_intent_failures:
        test = next(t for t in test_cases if t['id'] == result['id'])
        if 'MULTI_INTENT' in test.get('expected_intent', ''):
            recommendations.append(
                "📌 Multi-intent tests failing: Ensure intent detection counts ALL flags, "
                "not just first match"
            )
            break
    
    # Check for negation failures
    negation_failures = [r for r in results if r['status'] == 'FAIL']
    for result in negation_failures:
        test = next(t for t in test_cases if t['id'] == result['id'])
        if 'without' in test['query'].lower() or "don't" in test['query'].lower():
            recommendations.append(
                "📌 Negation tests failing: Add more negation keywords "
                "(\"without\", \"don't\", \"skip\", \"avoid\", \"no\")"
            )
            break
    
    # Check for feature engineering failures
    fe_failures = [r for r in results if r['status'] == 'FAIL']
    for result in fe_failures:
        test = next(t for t in test_cases if t['id'] == result['id'])
        if 'FEATURE_ENGINEERING' in test.get('expected_intent', ''):
            recommendations.append(
                "📌 Feature engineering tests failing: Check if 'time-based', 'engineer', "
                "'create features' keywords are detected"
            )
            break
    
    if recommendations:
        print("\nActionable Fixes:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    else:
        print("\n  ✅ No specific recommendations - all tests passed or issues are edge cases")
    
    # Success insights
    print("\n✨ SUCCESS INSIGHTS:")
    print("-" * 100)
    
    passed_tests = [r for r in results if r['status'] == 'PASS']
    if passed_tests:
        print(f"\n{len(passed_tests)} tests passed successfully!")
        
        # Find fastest tests
        if passed_tests:
            fastest = min(passed_tests, key=lambda x: x['time'])
            test = next(t for t in test_cases if t['id'] == fastest['id'])
            print(f"\n  ⚡ Fastest: {fastest['id']} ({fastest['time']:.2f}s) - {test['category']}")
    
    print("\n" + "="*100)
    
    return {
        "failed_by_category": failed_by_category,
        "failed_by_intent": failed_by_intent,
        "recommendations": recommendations
    }


# Example usage in test script:
"""
# At the end of test_robust_intent.py, add:

from test_analyzer import analyze_test_results

# After running all tests:
analysis = analyze_test_results(results, test_cases)

# This will show:
# - Which categories fail most (Simple vs Multi-Intent vs Edge cases)
# - Which intent types struggle (VISUALIZATION_ONLY vs MULTI_INTENT)
# - Performance bottlenecks (slow tests)
# - Actionable recommendations to fix failures
"""

if __name__ == "__main__":
    print("\n" + "="*100)
    print("📊 TEST RESULTS ANALYZER")
    print("="*100)
    print("\nThis module analyzes test results to identify patterns and provide fixes.")
    print("\nUsage:")
    print("```python")
    print("from test_analyzer import analyze_test_results")
    print("")
    print("# After running tests:")
    print("analysis = analyze_test_results(results, test_cases)")
    print("```")
    print("\nFeatures:")
    print("  ✓ Categorize failures by test type")
    print("  ✓ Identify performance bottlenecks")
    print("  ✓ Provide actionable recommendations")
    print("  ✓ Highlight success patterns")
    print("\n" + "="*100 + "\n")
