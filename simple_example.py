"""
Simple Example: Using the AI Agent Tools
=========================================

This demonstrates the most common workflows using the data science tools.
All examples use test_data/sample.csv (15 rows, 4 columns)
"""

import sys
sys.path.append('src')

def main():
    print("=" * 70)
    print("🤖 AI Agent Data Scientist - Simple Examples")
    print("=" * 70)
    
    # Example 1: Profile a dataset
    example_1_profiling()
    
    # Example 2: Train a classification model
    example_2_train_model()
    
    # Example 3: Analyze feature importance
    # example_3_feature_importance()  # Uncomment to run
    
    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   1. Check TOOLS_REFERENCE.md for all 44 tools")
    print("   2. Try: py src/cli.py profile your_data.csv")
    print("   3. Explore src/tools/ directory for all available tools")


def example_1_profiling():
    """Example 1: Profile a dataset to understand its structure."""
    print("\n📊 Example 1: Dataset Profiling")
    print("-" * 70)
    
    from tools.data_profiling import profile_dataset
    
    result = profile_dataset("test_data/sample.csv")
    
    print(f"✅ Dataset profiled successfully!")
    print(f"   • Rows: {result['shape']['rows']}")
    print(f"   • Columns: {result['shape']['columns']}")
    print(f"   • Memory: {result['memory_usage']:.2f} MB")
    print(f"   • Missing values: {result['overall_stats']['total_nulls']} ({result['overall_stats']['null_percentage']:.1f}%)")
    print(f"   • Duplicates: {result['overall_stats']['duplicate_rows']} ({result['overall_stats']['duplicate_percentage']:.1f}%)")
    
    print(f"\n   Column types:")
    print(f"      - Numeric: {len(result['column_types']['numeric'])} columns")
    print(f"      - Categorical: {len(result['column_types']['categorical'])} columns")
    print(f"      - Datetime: {len(result['column_types']['datetime'])} columns")


def example_2_train_model():
    """Example 2: Train a classification model."""
    print("\n🎯 Example 2: Train Classification Model")
    print("-" * 70)
    
    from tools.advanced_training import train_classification_model
    
    result = train_classification_model(
        file_path="test_data/sample.csv",
        target_column="purchased",
        model_type="random_forest",  # Options: random_forest, logistic_regression, svm, xgboost
        test_size=0.3
    )
    
    print(f"✅ Model trained successfully!")
    print(f"   • Model type: Random Forest")
    print(f"   • Accuracy: {result['metrics']['accuracy']:.3f}")
    print(f"   • Precision: {result['metrics']['precision']:.3f}")
    print(f"   • Recall: {result['metrics']['recall']:.3f}")
    print(f"   • F1 Score: {result['metrics']['f1']:.3f}")
    
    print(f"\n   Feature importance:")
    for i, (feature, importance) in enumerate(result['feature_importance'][:3], 1):
        print(f"      {i}. {feature}: {importance:.4f}")


def example_3_feature_importance():
    """Example 3: Analyze feature importance with SHAP."""
    print("\n📈 Example 3: Feature Importance Analysis")
    print("-" * 70)
    
    from tools.preprocessing import split_dataset
    from tools.advanced_training import train_classification_model
    from tools.analysis_diagnostics import analyze_feature_importance
    
    # First, split the data
    splits = split_dataset(
        file_path="test_data/sample.csv",
        target_column="purchased",
        test_size=0.3
    )
    
    # Train a model
    model_result = train_classification_model(
        file_path="test_data/sample.csv",
        target_column="purchased",
        model_type="random_forest"
    )
    
    # Analyze feature importance
    importance = analyze_feature_importance(
        model=model_result['model'],
        X=splits['X_train'],
        feature_names=splits['X_train'].columns
    )
    
    print(f"✅ Feature importance analyzed!")
    print(f"   Top 3 features:")
    for i, (feature, score) in enumerate(importance['feature_importance'][:3], 1):
        print(f"      {i}. {feature}: {score:.4f}")


if __name__ == "__main__":
    main()
