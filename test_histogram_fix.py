"""
Test: Verify agent generates HISTOGRAM (not scatter plot) when user asks for histogram
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import asyncio
import time

async def test_histogram_detection():
    """Test that 'plot a histogram' generates histogram, not scatter plot"""
    
    print("=" * 80)
    print("🧪 TEST: Histogram Detection")
    print("=" * 80)
    print("Request: 'Plot a histogram of Magnitude vs latitude'")
    print("Expected: generate_interactive_histogram (NOT generate_interactive_scatter)")
    print()
    
    agent = DataScienceCopilot()
    
    # Track tool calls
    tools_called = []
    
    # Hook into the agent to capture tool calls
    original_run = agent.run_analysis
    
    async def monitored_run(*args, **kwargs):
        result = await original_run(*args, **kwargs)
        return result
    
    agent.run_analysis = monitored_run
    
    start_time = time.time()
    
    try:
        result = await agent.run_analysis(
            file_path="./data/earthquake.csv",
            task_description="Plot a histogram of Magnitude vs latitude",
            target_col=None
        )
        
        execution_time = time.time() - start_time
        
        # Extract tool calls from result
        if "tools_executed" in result:
            tools_called = result["tools_executed"]
        
        print("\n" + "=" * 80)
        print("📊 RESULTS")
        print("=" * 80)
        
        print(f"✓ Tools Called: {len(tools_called)}")
        for i, tool in enumerate(tools_called, 1):
            print(f"  {i}. {tool}")
        
        print(f"✓ Execution Time: {execution_time:.2f}s")
        
        # Validation
        print("\n" + "=" * 80)
        print("✅ VALIDATION")
        print("=" * 80)
        
        has_histogram = "generate_interactive_histogram" in str(tools_called)
        has_scatter = "generate_interactive_scatter" in str(tools_called)
        
        if has_histogram and not has_scatter:
            print("✅ PASS: Generated histogram (correct!)")
        elif has_scatter and not has_histogram:
            print("❌ FAIL: Generated scatter plot instead of histogram")
            print("   Issue: Agent ignored 'histogram' keyword in request")
        elif has_histogram and has_scatter:
            print("⚠️ WARNING: Generated both histogram AND scatter")
        else:
            print("❌ FAIL: No visualization tool called")
        
        # Check if agent ran unnecessary tools
        unnecessary_tools = ["profile_dataset", "clean_missing_values", "train_baseline_models", 
                            "hyperparameter_tuning", "encode_categorical"]
        
        found_unnecessary = [tool for tool in tools_called if any(ut in str(tool) for ut in unnecessary_tools)]
        
        if found_unnecessary:
            print(f"⚠️ WARNING: Ran {len(found_unnecessary)} unnecessary tools: {found_unnecessary}")
        else:
            print("✅ PASS: No unnecessary tools executed")
        
        return {
            "test": "Histogram Detection",
            "passed": has_histogram and not has_scatter,
            "tools_called": len(tools_called),
            "execution_time": execution_time
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "test": "Histogram Detection",
            "passed": False,
            "error": str(e)
        }

async def test_scatter_vs_histogram():
    """Test that agent can distinguish between scatter and histogram requests"""
    
    print("\n\n" + "=" * 80)
    print("🧪 TEST: Scatter vs Histogram Distinction")
    print("=" * 80)
    
    test_cases = [
        {
            "request": "Plot a histogram of Magnitude",
            "expected_tool": "generate_interactive_histogram",
            "wrong_tool": "generate_interactive_scatter"
        },
        {
            "request": "Create a scatter plot of Magnitude vs Depth",
            "expected_tool": "generate_interactive_scatter",
            "wrong_tool": "generate_interactive_histogram"
        },
        {
            "request": "Show distribution of Magnitude",
            "expected_tool": "generate_interactive_histogram",
            "wrong_tool": "generate_interactive_scatter"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}/3 ---")
        print(f"Request: '{test_case['request']}'")
        print(f"Expected: {test_case['expected_tool']}")
        
        agent = DataScienceCopilot()
        
        try:
            result = await agent.run_analysis(
                file_path="./data/earthquake.csv",
                task_description=test_case["request"],
                target_col=None
            )
            
            tools_called = result.get("tools_executed", [])
            
            has_expected = test_case["expected_tool"] in str(tools_called)
            has_wrong = test_case["wrong_tool"] in str(tools_called)
            
            if has_expected and not has_wrong:
                print(f"✅ PASS: Called {test_case['expected_tool']}")
                results.append(True)
            elif has_wrong:
                print(f"❌ FAIL: Called {test_case['wrong_tool']} instead")
                results.append(False)
            else:
                print(f"⚠️ WARNING: Called neither expected nor wrong tool")
                results.append(False)
            
            # Rate limiting
            await asyncio.sleep(3.5)
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append(False)
    
    print("\n" + "=" * 80)
    print("📊 OVERALL RESULTS")
    print("=" * 80)
    print(f"Passed: {sum(results)}/{len(results)} tests")
    print(f"Pass Rate: {sum(results)/len(results)*100:.0f}%")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting Histogram Detection Tests...\n")
    
    # Run main test
    result1 = asyncio.run(test_histogram_detection())
    
    print("\n\n" + "=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)
    
    if result1.get("passed"):
        print("✓ Agent now correctly generates histograms when requested")
    else:
        print("✗ Agent still has issues with plot type detection")
