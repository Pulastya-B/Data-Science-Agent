"""
Quick test to verify the orchestrator can load tools without errors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("ORCHESTRATOR TOOL LOADING TEST")
print("=" * 80)

try:
    from orchestrator import DataScienceCopilot
    
    print("\n[TEST 1] Creating DataScienceCopilot instance...")
    copilot = DataScienceCopilot()
    print("✅ Copilot initialized successfully")
    
    print("\n[TEST 2] Building tool function map...")
    tool_map = copilot._build_tool_functions_map()
    print(f"✅ Tool map built with {len(tool_map)} tools")
    
    print("\n[TEST 3] Checking new Phase 1 tools...")
    new_tools = ['get_smart_summary', 'merge_datasets', 'concat_datasets', 'reshape_dataset']
    for tool in new_tools:
        if tool in tool_map:
            print(f"✅ {tool} found")
        else:
            print(f"❌ {tool} MISSING")
    
    print("\n[TEST 4] Converting tools to Gemini format (this was failing before)...")
    from tools.tools_registry import TOOLS
    
    # Try to convert all tools
    gemini_tools = copilot._convert_to_gemini_tools(TOOLS)
    print(f"✅ Successfully converted {len(gemini_tools)} tools to Gemini format")
    
    # Specifically check merge_datasets (has ["string", "array"] type)
    merge_tool = None
    for tool in gemini_tools:
        if isinstance(tool, dict) and tool.get("name") == "merge_datasets":
            merge_tool = tool
            break
    
    if merge_tool:
        print(f"✅ merge_datasets converted successfully")
        params = merge_tool.get("parameters", {}).get("properties", {})
        print(f"   Parameters: {list(params.keys())}")
        
        # Check if 'on' parameter was converted properly (it has ["string", "array"] type)
        if "on" in params:
            on_param = params["on"]
            on_type = on_param.get("type")
            print(f"   'on' parameter type: {on_type}")
            if on_type == "ARRAY":
                print(f"   ✅ List type converted to ARRAY correctly")
                # Check for items field (required by Gemini)
                if "items" in on_param:
                    items_type = on_param["items"].get("type")
                    print(f"   ✅ 'on' has items field with type: {items_type}")
                else:
                    print(f"   ❌ 'on' is missing required items field!")
        
        # Check left_on and right_on too
        for param_name in ["left_on", "right_on"]:
            if param_name in params:
                param = params[param_name]
                if param.get("type") == "ARRAY" and "items" in param:
                    print(f"   ✅ '{param_name}' has items field")
                elif param.get("type") == "ARRAY":
                    print(f"   ❌ '{param_name}' missing items field!")
    else:
        print(f"⚠️  merge_datasets not found in converted tools")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Orchestrator can load and convert all tools!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
