"""
Example: Using AI Agent Data Scientist Tools Directly
======================================================

This script demonstrates how to use the 44 data science tools
directly in your Python code without relying on the AI-powered CLI.
"""

import sys
sys.path.append('src')

from tools.data_profiling import profile_dataset
from tools.data_cleaning import handle_missing_data, encode_categorical
from tools.advanced_training import train_classification_model, tune_hyperparameters
from tools.preprocessing import split_dataset
from tools.analysis_diagnostics import analyze_feature_importance

def example_workflow():
    """Complete data science workflow using the tools."""
    
    print("=" * 60)
    print("🤖 AI Agent Data Scientist - Example Workflow")
    print("=" * 60)
    
    # Step 1: Profile the dataset
    print("\n📊 Step 1: Profiling dataset...")
    profile = profile_dataset("test_data/sample.csv")
    print(f"   ✅ Dataset has {profile['shape']['rows']} rows and {profile['shape']['columns']} columns")
    
    # Step 2: Handle missing data (if any)
    print("\n🧹 Step 2: Cleaning data...")
    cleaned = handle_missing_data(
        "test_data/sample.csv",
        method="drop"  # Can also use: mean, median, mode, knn, interpolate
    )
    print(f"   ✅ Data cleaned")
    
    # Step 3: Split dataset
    print("\n🔀 Step 3: Splitting dataset...")
    splits = split_dataset(
        "test_data/sample.csv",
        target_column="purchased",
        test_size=0.3
    )
    print(f"   ✅ Train: {splits['X_train'].shape[0]} rows")
    print(f"   ✅ Test: {splits['X_test'].shape[0]} rows")
    
    # Step 4: Train a model
    print("\n🎯 Step 4: Training classification model...")
    model_result = train_classification_model(
        "test_data/sample.csv",
        target_column="purchased",
        model_type="random_forest",
        test_size=0.3
    )
    print(f"   ✅ Model trained!")
    print(f"   ✅ Accuracy: {model_result['metrics']['accuracy']:.3f}")
    print(f"   ✅ Precision: {model_result['metrics']['precision']:.3f}")
    print(f"   ✅ Recall: {model_result['metrics']['recall']:.3f}")
    print(f"   ✅ F1: {model_result['metrics']['f1']:.3f}")
    
    # Step 5: Analyze feature importance
    print("\n📈 Step 5: Analyzing feature importance...")
    importance = analyze_feature_importance(
        model_result['model'],
        splits['X_train'],
        feature_names=splits['X_train'].columns
    )
    print(f"   ✅ Top features:")
    for i, (feature, score) in enumerate(importance['feature_importance'][:3], 1):
        print(f"      {i}. {feature}: {score:.4f}")
    
    # Step 6: Hyperparameter tuning (optional - can be slow)
    print("\n⚙️  Step 6: Hyperparameter tuning (optional)...")
    print("   ℹ️  Skipping to save time. To enable, uncomment the code below.")
    # tuned = tune_hyperparameters(
    #     "test_data/sample.csv",
    #     target_column="purchased",
    #     model_type="random_forest",
    #     n_trials=10  # Use more trials for better results
    # )
    # print(f"   ✅ Best score: {tuned['best_score']:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ Workflow complete!")
    print("=" * 60)
    print("\n💡 Tips:")
    print("   - Check TOOLS_REFERENCE.md for all 44 tools")
    print("   - Each tool returns structured data you can use")
    print("   - Tools work with CSV files or Polars DataFrames")
    print("   - Combine tools to build custom workflows")
    

def example_nlp_workflow():
    """Example NLP workflow."""
    print("\n" + "=" * 60)
    print("📝 NLP Example (requires text data)")
    print("=" * 60)
    
    from tools.nlp_text_analytics import (
        perform_sentiment_analysis,
        extract_keywords,
        perform_topic_modeling
    )
    
    # Create sample text data
    import polars as pl
    text_data = pl.DataFrame({
        "text": [
            "This product is amazing! I love it so much.",
            "Terrible experience. Would not recommend.",
            "It's okay, nothing special.",
            "Great quality and fast shipping!",
            "Disappointed with the purchase."
        ]
    })
    text_data.write_csv("test_data/sample_text.csv")
    
    print("\n📊 Analyzing sentiment...")
    sentiment = perform_sentiment_analysis("test_data/sample_text.csv", "text")
    print(f"   ✅ Analyzed {len(sentiment['sentiments'])} texts")
    print(f"   ✅ Average polarity: {sum(s['polarity'] for s in sentiment['sentiments']) / len(sentiment['sentiments']):.3f}")
    
    print("\n🔑 Extracting keywords...")
    keywords = extract_keywords("test_data/sample_text.csv", "text", top_n=5)
    print(f"   ✅ Top keywords: {', '.join(keywords['keywords'][:5])}")
    
    print("\n📚 Topic modeling...")
    topics = perform_topic_modeling(
        "test_data/sample_text.csv",
        text_column="text",
        n_topics=2,
        method="lda"
    )
    print(f"   ✅ Identified {topics['n_topics']} topics")
    for i, topic in enumerate(topics['topics'], 1):
        print(f"      Topic {i}: {', '.join(topic['words'][:5])}")


def example_time_series_workflow():
    """Example time series workflow."""
    print("\n" + "=" * 60)
    print("📈 Time Series Example (requires date column)")
    print("=" * 60)
    
    from tools.time_series_forecasting import (
        forecast_timeseries,
        detect_seasonality,
        decompose_timeseries
    )
    
    # Create sample time series data
    import polars as pl
    from datetime import datetime, timedelta
    
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
    values = [100 + i * 2 + (i % 7) * 5 for i in range(30)]
    
    ts_data = pl.DataFrame({
        "date": dates,
        "value": values
    })
    ts_data.write_csv("test_data/sample_timeseries.csv")
    
    print("\n🔍 Detecting seasonality...")
    seasonality = detect_seasonality("test_data/sample_timeseries.csv", "date", "value")
    print(f"   ✅ Seasonal: {seasonality['is_seasonal']}")
    print(f"   ✅ Period: {seasonality['period']} days")
    
    print("\n📊 Decomposing time series...")
    decomp = decompose_timeseries("test_data/sample_timeseries.csv", "date", "value")
    print(f"   ✅ Trend detected")
    print(f"   ✅ Seasonal component extracted")
    
    print("\n🔮 Forecasting...")
    forecast = forecast_timeseries(
        "test_data/sample_timeseries.csv",
        date_column="date",
        value_column="value",
        periods=7  # Forecast next 7 days
    )
    print(f"   ✅ Generated {len(forecast['forecast'])} predictions")


def example_business_intelligence():
    """Example BI workflow."""
    print("\n" + "=" * 60)
    print("💼 Business Intelligence Example")
    print("=" * 60)
    
    from tools.business_intelligence import (
        calculate_rfm_segments,
        analyze_customer_lifetime_value
    )
    
    # Create sample customer data
    import polars as pl
    from datetime import datetime, timedelta
    
    customers = pl.DataFrame({
        "customer_id": [1, 2, 3, 4, 5] * 3,
        "purchase_date": [datetime(2023, 1, 1) + timedelta(days=i*10) for i in range(15)],
        "amount": [100, 200, 150, 300, 250, 120, 180, 220, 190, 280, 140, 210, 170, 240, 200]
    })
    customers.write_csv("test_data/sample_customers.csv")
    
    print("\n📊 Calculating RFM segments...")
    rfm = calculate_rfm_segments(
        "test_data/sample_customers.csv",
        customer_id="customer_id",
        date_column="purchase_date",
        amount_column="amount"
    )
    print(f"   ✅ Segmented {len(rfm['segments'])} customers")
    print(f"   ✅ Top segment: {rfm['segments'][0]['segment']}")
    
    print("\n💰 Analyzing customer lifetime value...")
    clv = analyze_customer_lifetime_value(
        "test_data/sample_customers.csv",
        customer_id="customer_id",
        date_column="purchase_date",
        amount_column="amount"
    )
    print(f"   ✅ Average CLV: ${clv['avg_clv']:.2f}")
    print(f"   ✅ Total revenue: ${clv['total_revenue']:.2f}")


if __name__ == "__main__":
    # Run the main workflow
    example_workflow()
    
    # Uncomment to try other examples:
    # example_nlp_workflow()
    # example_time_series_workflow()
    # example_business_intelligence()
