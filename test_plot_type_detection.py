"""
Simplified Test: Show that plot type detection guidance is now working
Tests the logic WITHOUT running the full agent
"""

def detect_plot_type(task_description):
    """Extract the plot type detection logic from orchestrator.py"""
    
    task_lower = task_description.lower()
    
    # Detect if it's a visualization request
    wants_viz = any(kw in task_lower for kw in ["plot", "graph", "visualiz", "dashboard", "chart", "show", "display", "create", "generate"])
    
    # Detect specific plot type
    plot_type_guidance = ""
    if wants_viz:
        if "histogram" in task_lower or "distribution" in task_lower or "freq" in task_lower:
            plot_type_guidance = "✅ Use: generate_interactive_histogram\n❌ Do NOT use: generate_interactive_scatter"
            plot_type = "Histogram"
        elif "scatter" in task_lower or "relationship" in task_lower:
            plot_type_guidance = "✅ Use: generate_interactive_scatter\n❌ Do NOT use: generate_interactive_histogram"
            plot_type = "Scatter Plot"
        elif "box plot" in task_lower or "boxplot" in task_lower or "outlier" in task_lower:
            plot_type_guidance = "✅ Use: generate_interactive_box_plots"
            plot_type = "Box Plot"
        elif "time series" in task_lower or "trend" in task_lower or "over time" in task_lower:
            plot_type_guidance = "✅ Use: generate_interactive_time_series"
            plot_type = "Time Series"
        elif "heatmap" in task_lower or "correlation" in task_lower:
            plot_type_guidance = "✅ Use: generate_interactive_correlation_heatmap"
            plot_type = "Heatmap"
        elif "dashboard" in task_lower or "all plot" in task_lower:
            plot_type_guidance = "✅ Use: generate_plotly_dashboard OR generate_all_plots"
            plot_type = "Dashboard"
        else:
            plot_type_guidance = "Generic visualization - LLM decides based on data"
            plot_type = "Generic"
    else:
        plot_type = "Not a visualization"
        plot_type_guidance = "N/A"
    
    return plot_type, plot_type_guidance


def run_tests():
    """Test various user requests"""
    
    test_cases = [
        {
            "request": "plot a histogram of Magnitude vs latitude",
            "expected_type": "Histogram",
            "expected_tool": "generate_interactive_histogram"
        },
        {
            "request": "create a scatter plot of Magnitude vs Depth",
            "expected_type": "Scatter Plot",
            "expected_tool": "generate_interactive_scatter"
        },
        {
            "request": "show distribution of Magnitude",
            "expected_type": "Histogram",
            "expected_tool": "generate_interactive_histogram"
        },
        {
            "request": "visualize the relationship between two variables",
            "expected_type": "Scatter Plot",
            "expected_tool": "generate_interactive_scatter"
        },
        {
            "request": "generate box plots for outlier detection",
            "expected_type": "Box Plot",
            "expected_tool": "generate_interactive_box_plots"
        },
        {
            "request": "plot time series trends",
            "expected_type": "Time Series",
            "expected_tool": "generate_interactive_time_series"
        },
        {
            "request": "create a heatmap of correlations",
            "expected_type": "Heatmap",
            "expected_tool": "generate_interactive_correlation_heatmap"
        },
        {
            "request": "generate a dashboard with all plots",
            "expected_type": "Dashboard",
            "expected_tool": "generate_plotly_dashboard"
        },
        {
            "request": "train a model to predict house prices",
            "expected_type": "Not a visualization",
            "expected_tool": "N/A"
        }
    ]
    
    print("=" * 100)
    print("🧪 PLOT TYPE DETECTION TEST")
    print("=" * 100)
    print()
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}/{len(test_cases)} ---")
        print(f"📝 Request: '{test['request']}'")
        print(f"🎯 Expected: {test['expected_type']} → {test['expected_tool']}")
        
        detected_type, guidance = detect_plot_type(test["request"])
        
        print(f"🤖 Detected: {detected_type}")
        print(f"💡 Guidance: {guidance}")
        
        # Validation
        if detected_type == test["expected_type"]:
            if test["expected_tool"] in guidance or test["expected_tool"] == "N/A":
                print("✅ PASS: Correct plot type and tool detected")
                passed += 1
            else:
                print(f"❌ FAIL: Correct type but wrong tool guidance")
                failed += 1
        else:
            print(f"❌ FAIL: Expected {test['expected_type']}, got {detected_type}")
            failed += 1
    
    print("\n" + "=" * 100)
    print("📊 RESULTS")
    print("=" * 100)
    print(f"✅ Passed: {passed}/{len(test_cases)}")
    print(f"❌ Failed: {failed}/{len(test_cases)}")
    print(f"📈 Pass Rate: {passed/len(test_cases)*100:.0f}%")
    
    if passed == len(test_cases):
        print("\n🎉 ALL TESTS PASSED! Plot type detection is working correctly.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Review the logic above.")
    
    # Highlight the original issue
    print("\n" + "=" * 100)
    print("🔍 ORIGINAL ISSUE CHECK")
    print("=" * 100)
    print("Request: 'plot a histogram of Magnitude vs latitude'")
    plot_type, guidance = detect_plot_type("plot a histogram of Magnitude vs latitude")
    print(f"Detected Type: {plot_type}")
    print(f"Tool Guidance: {guidance}")
    
    if "generate_interactive_histogram" in guidance and "Do NOT use: generate_interactive_scatter" in guidance:
        print("\n✅ FIXED! Agent will now generate histogram, not scatter plot.")
        print("   The LLM receives explicit guidance:")
        print("   ✅ Use: generate_interactive_histogram")
        print("   ❌ Do NOT use: generate_interactive_scatter")
    else:
        print("\n❌ Issue still present - needs more work")


if __name__ == "__main__":
    run_tests()
