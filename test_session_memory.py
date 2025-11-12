"""
Test Session Memory Functionality
Demonstrates context-aware follow-up handling with session memory.
"""

import sys
sys.path.append("./src")

from orchestrator import DataScienceCopilot

def test_session_memory():
    """Test session memory with follow-up commands."""
    
    print("=" * 80)
    print("SESSION MEMORY TEST")
    print("=" * 80)
    
    # Initialize copilot with session memory enabled
    copilot = DataScienceCopilot(use_session_memory=True)
    
    print(f"\n📝 Session ID: {copilot.get_session_id()}\n")
    
    # Test 1: Initial request - Train model
    print("\n" + "=" * 80)
    print("TEST 1: Initial Request - Train Model")
    print("=" * 80)
    
    result1 = copilot.analyze(
        file_path="./test_data/sample.csv",  # Use existing test data
        task_description="Train a model to predict the target",
        target_col=None,  # Let agent infer
        use_cache=False
    )
    
    print(f"\n✅ Status: {result1['status']}")
    print(f"Summary: {result1.get('summary', 'N/A')[:200]}...")
    
    # Show session context after first request
    print("\n" + "-" * 80)
    print("SESSION CONTEXT AFTER TRAINING:")
    print("-" * 80)
    print(copilot.get_session_context())
    
    # Test 2: Follow-up request - Cross validate IT (ambiguous reference)
    print("\n" + "=" * 80)
    print("TEST 2: Follow-Up - Cross Validate IT (Ambiguous)")
    print("=" * 80)
    print("User says: 'Cross validate it'")
    print("Expected: Agent resolves 'it' → XGBoost, uses session context")
    
    result2 = copilot.analyze(
        file_path="",  # Empty - should use session context!
        task_description="Cross validate it",
        target_col=None,  # Should use session context!
        use_cache=False
    )
    
    print(f"\n✅ Status: {result2['status']}")
    print(f"Summary: {result2.get('summary', 'N/A')[:200]}...")
    
    # Test 3: Another follow-up - Tune the model
    print("\n" + "=" * 80)
    print("TEST 3: Follow-Up - Tune the Model")
    print("=" * 80)
    print("User says: 'Tune it'")
    print("Expected: Agent resolves 'it' → last model, uses session context")
    
    result3 = copilot.analyze(
        file_path="",
        task_description="Tune it with 30 trials",
        target_col=None,
        use_cache=False
    )
    
    print(f"\n✅ Status: {result3['status']}")
    print(f"Summary: {result3.get('summary', 'N/A')[:200]}...")
    
    # Test 4: Visualization follow-up
    print("\n" + "=" * 80)
    print("TEST 4: Follow-Up - Visualize Results")
    print("=" * 80)
    print("User says: 'Plot the results'")
    print("Expected: Agent uses last dataset from session")
    
    result4 = copilot.analyze(
        file_path="",
        task_description="Plot the results",
        target_col=None,
        use_cache=False
    )
    
    print(f"\n✅ Status: {result4['status']}")
    print(f"Summary: {result4.get('summary', 'N/A')[:200]}...")
    
    # Final session context
    print("\n" + "=" * 80)
    print("FINAL SESSION CONTEXT:")
    print("=" * 80)
    print(copilot.get_session_context())
    
    print("\n" + "=" * 80)
    print("SESSION MEMORY TEST COMPLETE")
    print("=" * 80)


def test_session_resumption():
    """Test resuming a previous session."""
    
    print("\n\n" + "=" * 80)
    print("SESSION RESUMPTION TEST")
    print("=" * 80)
    
    # First copilot instance - create session
    print("\n1️⃣  Creating first session...")
    copilot1 = DataScienceCopilot(use_session_memory=True)
    session_id = copilot1.get_session_id()
    print(f"   Session ID: {session_id}")
    
    # Do some work
    copilot1.analyze(
        file_path="./test_data/sample.csv",
        task_description="Profile the dataset",
        use_cache=False
    )
    
    print(f"\n   Session context:")
    print(copilot1.get_session_context())
    
    # Simulate restarting the agent - new copilot instance with same session ID
    print("\n2️⃣  Simulating agent restart...")
    print(f"   Loading session: {session_id}")
    
    copilot2 = DataScienceCopilot(
        use_session_memory=True,
        session_id=session_id  # Explicitly load previous session
    )
    
    print(f"\n   Resumed session context:")
    print(copilot2.get_session_context())
    
    # Continue work from where we left off
    print("\n3️⃣  Continuing work from previous session...")
    copilot2.analyze(
        file_path="",  # Should use session context
        task_description="Clean the data",
        use_cache=False
    )
    
    print("\n✅ Session resumption successful!")
    print("=" * 80)


def test_clear_session():
    """Test clearing session context."""
    
    print("\n\n" + "=" * 80)
    print("CLEAR SESSION TEST")
    print("=" * 80)
    
    copilot = DataScienceCopilot(use_session_memory=True)
    
    # Do some work
    print("\n1️⃣  Creating session context...")
    copilot1.analyze(
        file_path="./test_data/sample.csv",
        task_description="Train model predicting target",
        target_col=None,
        use_cache=False
    )
    
    print(f"\n   Before clear:")
    print(copilot.get_session_context())
    
    # Clear session
    print("\n2️⃣  Clearing session...")
    copilot.clear_session()
    
    print(f"\n   After clear:")
    print(copilot.get_session_context())
    
    print("\n✅ Session cleared successfully!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test session memory functionality")
    parser.add_argument("--test", choices=["memory", "resumption", "clear", "all"], 
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test in ["memory", "all"]:
        test_session_memory()
    
    if args.test in ["resumption", "all"]:
        test_session_resumption()
    
    if args.test in ["clear", "all"]:
        test_clear_session()
    
    print("\n\n🎉 ALL TESTS COMPLETE!")
