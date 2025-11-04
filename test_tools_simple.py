"""
Simple test script to verify individual tools work
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all tool modules can be imported"""
    print("Testing tool imports...")
    
    try:
        from tools.data_profiling import profile_dataset
        print("✅ data_profiling imported successfully")
    except Exception as e:
        print(f"❌ data_profiling import failed: {e}")
        return False
    
    try:
        from tools.advanced_training import hyperparameter_tuning
        print("✅ advanced_training imported successfully")
    except Exception as e:
        print(f"❌ advanced_training import failed: {e}")
        return False
    
    try:
        from tools.nlp_text_analytics import perform_topic_modeling
        print("✅ nlp_text_analytics imported successfully")
    except Exception as e:
        print(f"❌ nlp_text_analytics import failed: {e}")
        return False
    
    try:
        from tools.business_intelligence import perform_rfm_analysis
        print("✅ business_intelligence imported successfully")
    except Exception as e:
        print(f"❌ business_intelligence import failed: {e}")
        return False
    
    try:
        from tools.computer_vision import extract_image_features
        print("✅ computer_vision imported successfully")
    except Exception as e:
        print(f"❌ computer_vision import failed: {e}")
        return False
    
    print("\n✅ All tools imported successfully!\n")
    return True


def test_simple_profiling():
    """Test basic data profiling"""
    print("Testing data profiling with sample data...")
    
    try:
        import polars as pl
        from tools.data_profiling import profile_dataset
        
        # Create simple test data
        data = pl.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'income': [50000, 60000, 75000, 90000, 100000],
            'score': [75, 82, 88, 92, 85],
            'purchased': [1, 1, 1, 1, 0]
        })
        
        print(f"Test data shape: {data.shape}")
        
        # Profile the data
        result = profile_dataset(data)
        
        print("\n✅ Profiling successful!")
        print(f"   Rows: {result['shape']['rows']}")
        print(f"   Columns: {result['shape']['columns']}")
        print(f"   Column types: {len(result['dtypes'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Profiling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_topic_modeling():
    """Test NLP topic modeling"""
    print("\nTesting topic modeling with sample text...")
    
    try:
        import polars as pl
        from tools.nlp_text_analytics import perform_topic_modeling
        
        # Create sample text data
        texts = [
            "Machine learning is fascinating and powerful",
            "Deep learning models are very effective",
            "Natural language processing helps understand text",
            "Computer vision can detect objects in images",
            "Data science involves statistics and programming"
        ] * 3  # Repeat to have enough data
        
        data = pl.DataFrame({'text': texts})
        
        print(f"Test data: {len(texts)} documents")
        
        # Perform topic modeling
        result = perform_topic_modeling(
            data=data,
            text_column='text',
            n_topics=2,
            method='lda',
            n_top_words=5
        )
        
        print("\n✅ Topic modeling successful!")
        print(f"   Found {len(result['topics'])} topics")
        for topic in result['topics']:
            print(f"   Topic {topic['topic_id']}: {', '.join(topic['words'][:3])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Topic modeling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Data Science AI Agent - Tool Testing")
    print("=" * 60)
    print()
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Check dependencies.")
        sys.exit(1)
    
    # Test 2: Simple profiling
    if not test_simple_profiling():
        print("\n❌ Profiling test failed.")
        sys.exit(1)
    
    # Test 3: Topic modeling
    if not test_topic_modeling():
        print("\n⚠️  Topic modeling test failed (may need optional dependencies)")
    
    print("\n" + "=" * 60)
    print("✅ Core tools are working! Agent is ready to use.")
    print("=" * 60)
