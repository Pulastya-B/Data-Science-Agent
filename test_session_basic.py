"""
Quick Session Memory Test
Tests basic session memory functionality without requiring full model training.
"""

import sys
sys.path.append("./src")

from orchestrator import DataScienceCopilot

def test_basic_session():
    """Test basic session memory initialization and context tracking."""
    
    print("=" * 80)
    print("BASIC SESSION MEMORY TEST")
    print("=" * 80)
    
    # Test 1: Initialize with session memory
    print("\n✅ Test 1: Initialize copilot with session memory")
    copilot = DataScienceCopilot(use_session_memory=True)
    
    session_id = copilot.get_session_id()
    print(f"   Session ID: {session_id}")
    assert session_id is not None, "Session ID should not be None"
    print("   PASS ✓")
    
    # Test 2: Get empty context
    print("\n✅ Test 2: Get session context (should be empty)")
    context = copilot.get_session_context()
    print(f"   Context: {context}")
    assert "No previous context" in context or "Session Context" in context
    print("   PASS ✓")
    
    # Test 3: Manually update session (simulate tool execution)
    print("\n✅ Test 3: Update session manually")
    copilot.session.update(
        last_dataset="./test_data/sample.csv",
        last_model="XGBoost",
        best_score=0.92,
        last_target_col="target"
    )
    
    context = copilot.get_session_context()
    print(f"   Updated Context:\n{context}")
    assert "XGBoost" in context
    assert "0.92" in context
    print("   PASS ✓")
    
    # Test 4: Test ambiguity resolution
    print("\n✅ Test 4: Resolve ambiguous request")
    resolved = copilot.session.resolve_ambiguity("Cross validate it")
    print(f"   Resolved parameters: {resolved}")
    assert resolved.get("model_type") == "xgboost"
    assert resolved.get("file_path") == "./test_data/sample.csv"
    assert resolved.get("target_col") == "target"
    print("   PASS ✓")
    
    # Test 5: Save session
    print("\n✅ Test 5: Save session to database")
    copilot.session_store.save(copilot.session)
    print("   Session saved")
    
    # Test 6: Load session
    print("\n✅ Test 6: Load session from database")
    loaded_session = copilot.session_store.load(session_id)
    assert loaded_session is not None
    assert loaded_session.last_model == "XGBoost"
    assert loaded_session.best_score == 0.92
    print(f"   Loaded: Model={loaded_session.last_model}, Score={loaded_session.best_score}")
    print("   PASS ✓")
    
    # Test 7: Clear session
    print("\n✅ Test 7: Clear session context")
    copilot.clear_session()
    context = copilot.get_session_context()
    print(f"   Context after clear: {context}")
    assert "No previous context" in context
    print("   PASS ✓")
    
    # Test 8: Session without memory
    print("\n✅ Test 8: Initialize without session memory")
    copilot_no_mem = DataScienceCopilot(use_session_memory=False)
    assert copilot_no_mem.session is None
    print("   Session disabled as expected")
    print("   PASS ✓")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓✓✓")
    print("=" * 80)
    print("\nSession Memory Features Verified:")
    print("  ✅ Session creation and ID generation")
    print("  ✅ Context tracking (dataset, model, score, target)")
    print("  ✅ Ambiguity resolution (pronouns → actual values)")
    print("  ✅ Session persistence (save/load from SQLite)")
    print("  ✅ Session clearing")
    print("  ✅ Optional session memory (can be disabled)")
    print("\n🎉 Session memory is working correctly!")


if __name__ == "__main__":
    try:
        test_basic_session()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
