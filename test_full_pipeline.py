"""
Test the full pipeline: compress registry -> convert to Gemini format
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("FULL PIPELINE TEST: Compress + Gemini Conversion")
print("=" * 80)

try:
    from orchestrator import DataScienceCopilot
    
    print("\n[TEST 1] Creating DataScienceCopilot...")
    copilot = DataScienceCopilot()
    print("✅ Copilot initialized")
    
    print("\n[TEST 2] Compressing tools registry (this is what UI uses)...")
    compressed_tools = copilot._compress_tools_registry()
    print(f"✅ Compressed {len(compressed_tools)} tools")
    
    # Find merge_datasets in compressed tools
    merge_tool_compressed = None
    for tool in compressed_tools:
        if tool["function"]["name"] == "merge_datasets":
            merge_tool_compressed = tool
            break
    
    if merge_tool_compressed:
        print(f"\n[TEST 3] Checking merge_datasets in compressed format...")
        params = merge_tool_compressed["function"]["parameters"]["properties"]
        print(f"   Parameters: {list(params.keys())}")
        
        # Check if on, left_on, right_on have items field
        for param_name in ["on", "left_on", "right_on"]:
            if param_name in params:
                param = params[param_name]
                param_type = param.get("type")
                has_items = "items" in param
                print(f"   {param_name}: type={param_type}, has_items={has_items}")
                
                if has_items:
                    print(f"      ✅ items field present: {param['items']}")
                else:
                    print(f"      ❌ items field MISSING!")
    else:
        print("❌ merge_datasets not found in compressed tools")
        sys.exit(1)
    
    print("\n[TEST 4] Converting compressed tools to Gemini format...")
    gemini_tools = copilot._convert_to_gemini_tools(compressed_tools)
    print(f"✅ Converted {len(gemini_tools)} tools to Gemini format")
    
    # Find merge_datasets in Gemini tools
    merge_tool_gemini = None
    for tool in gemini_tools:
        if isinstance(tool, dict) and tool.get("name") == "merge_datasets":
            merge_tool_gemini = tool
            break
    
    if merge_tool_gemini:
        print(f"\n[TEST 5] Checking merge_datasets in Gemini format...")
        params = merge_tool_gemini.get("parameters", {}).get("properties", {})
        
        all_good = True
        for param_name in ["on", "left_on", "right_on"]:
            if param_name in params:
                param = params[param_name]
                param_type = param.get("type")
                has_items = "items" in param
                
                if param_type == "ARRAY" and has_items:
                    items_type = param["items"].get("type")
                    print(f"   ✅ {param_name}: type=ARRAY, items.type={items_type}")
                elif param_type == "ARRAY" and not has_items:
                    print(f"   ❌ {param_name}: type=ARRAY but MISSING items field!")
                    all_good = False
                else:
                    print(f"   ⚠️  {param_name}: type={param_type}")
        
        if all_good:
            print(f"\n✅✅✅ ALL ARRAY PARAMETERS HAVE items FIELD!")
            print("=" * 80)
            print("✅ PIPELINE TEST PASSED - UI should work now!")
            print("=" * 80)
        else:
            print("\n❌ Some array parameters missing items field")
            sys.exit(1)
    else:
        print("❌ merge_datasets not found in Gemini tools")
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
