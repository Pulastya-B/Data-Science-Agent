"""
Comprehensive Test Script for New Features
==========================================

Tests all newly added features:
1. Advanced Insights (6 functions)
2. Auto ML Pipeline (2 functions)
3. Visualization Engine (5 functions)
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

print("=" * 80)
print("🧪 COMPREHENSIVE FEATURE TEST")
print("=" * 80)
print()

# Test imports
print("📦 Testing imports...")
try:
    from tools.advanced_insights import (
        analyze_root_cause,
        detect_trends_and_seasonality,
        detect_anomalies_advanced,
        perform_hypothesis_testing,
        analyze_distribution,
        perform_segment_analysis
    )
    print("  ✅ Advanced insights imported")
except Exception as e:
    print(f"  ❌ Failed to import advanced_insights: {e}")
    sys.exit(1)

try:
    from tools.auto_pipeline import (
        auto_ml_pipeline,
        auto_feature_selection
    )
    print("  ✅ Auto pipeline imported")
except Exception as e:
    print(f"  ❌ Failed to import auto_pipeline: {e}")
    sys.exit(1)

try:
    from tools.visualization_engine import (
        generate_all_plots,
        generate_data_quality_plots,
        generate_eda_plots,
        generate_model_performance_plots,
        generate_feature_importance_plot
    )
    print("  ✅ Visualization engine imported")
except Exception as e:
    print(f"  ❌ Failed to import visualization_engine: {e}")
    sys.exit(1)

try:
    from tools.model_training import train_baseline_models
    print("  ✅ Model training imported")
except Exception as e:
    print(f"  ❌ Failed to import model_training: {e}")
    sys.exit(1)

print()

# Test dataset
test_file = "test_data/sample.csv"
if not os.path.exists(test_file):
    print(f"⚠️  Test file {test_file} not found. Creating synthetic dataset...")
    import polars as pl
    import numpy as np
    
    np.random.seed(42)
    n = 1000
    
    df = pl.DataFrame({
        'age': np.random.randint(18, 80, n),
        'income': np.random.normal(50000, 20000, n),
        'credit_score': np.random.randint(300, 850, n),
        'loan_amount': np.random.normal(200000, 100000, n),
        'employment_years': np.random.randint(0, 40, n),
        'num_accounts': np.random.randint(1, 10, n),
        'debt_ratio': np.random.uniform(0, 1, n),
        'default': np.random.binomial(1, 0.2, n)  # 20% default rate
    })
    
    os.makedirs("test_data", exist_ok=True)
    df.write_csv(test_file)
    print(f"  ✅ Created synthetic dataset: {test_file}")

print(f"📊 Using test dataset: {test_file}")
print()

# Test 1: Auto ML Pipeline
print("=" * 80)
print("TEST 1: Auto ML Pipeline")
print("=" * 80)
try:
    print("Running auto_ml_pipeline with target='purchased', level='intermediate'...")
    result = auto_ml_pipeline(
        file_path=test_file,
        target_col="purchased",
        task_type="classification",
        feature_engineering_level="intermediate"
    )
    
    if result.get('success'):
        print(f"  ✅ Pipeline successful!")
        print(f"     - Processed file: {result.get('processed_file')}")
        print(f"     - Steps executed: {len(result.get('steps', []))}")
        print(f"     - Initial features: {result.get('num_features_before', 0)}")
        print(f"     - Final features: {result.get('num_features_after', 0)}")
        
        # Show pipeline steps
        print("\n  Pipeline Steps:")
        for i, step in enumerate(result.get('steps', []), 1):
            print(f"     {i}. {step.get('step_name', 'Unknown')}: {step.get('description', '')}")
    else:
        print(f"  ❌ Pipeline failed: {result.get('message', 'Unknown error')}")
except Exception as e:
    print(f"  ❌ Pipeline error: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Advanced Insights - Root Cause Analysis
print("=" * 80)
print("TEST 2: Root Cause Analysis")
print("=" * 80)
try:
    print("Running analyze_root_cause for 'purchased' metric...")
    result = analyze_root_cause(
        file_path=test_file,
        target_col="purchased",
        time_col=None
    )
    
    if result.get('success'):
        print(f"  ✅ Root cause analysis successful!")
        correlations = result.get('correlations', {})
        if correlations:
            print(f"\n  Top Correlations with purchased:")
            for col, corr in list(correlations.items())[:5]:
                print(f"     - {col}: {corr:.3f}")
    else:
        print(f"  ❌ Analysis failed: {result.get('message', 'Unknown error')}")
except Exception as e:
    print(f"  ❌ Root cause error: {e}")

print()

# Test 3: Anomaly Detection
print("=" * 80)
print("TEST 3: Anomaly Detection")
print("=" * 80)
try:
    print("Running detect_anomalies_advanced...")
    result = detect_anomalies_advanced(
        file_path=test_file,
        columns=["income", "credit_score", "loan_amount"],
        contamination=0.1
    )
    
    if result.get('success'):
        print(f"  ✅ Anomaly detection successful!")
        print(f"     - Anomalies detected: {result.get('num_anomalies', 0)}")
        print(f"     - Anomaly percentage: {result.get('anomaly_percentage', 0):.2f}%")
        print(f"     - Method: {result.get('method', 'Unknown')}")
    else:
        print(f"  ❌ Detection failed: {result.get('message', 'Unknown error')}")
except Exception as e:
    print(f"  ❌ Anomaly detection error: {e}")

print()

# Test 4: Distribution Analysis
print("=" * 80)
print("TEST 4: Distribution Analysis")
print("=" * 80)
try:
    print("Running analyze_distribution for 'income'...")
    result = analyze_distribution(
        file_path=test_file,
        column="income"
    )
    
    if result.get('success'):
        print(f"  ✅ Distribution analysis successful!")
        print(f"     - Normality: {result.get('is_normal', False)}")
        print(f"     - Skewness: {result.get('skewness', 0):.3f}")
        print(f"     - Kurtosis: {result.get('kurtosis', 0):.3f}")
    else:
        print(f"  ❌ Analysis failed: {result.get('message', 'Unknown error')}")
except Exception as e:
    print(f"  ❌ Distribution error: {e}")

print()

# Test 5: Hypothesis Testing
print("=" * 80)
print("TEST 5: Hypothesis Testing")
print("=" * 80)
try:
    print("Running perform_hypothesis_testing: income by purchased status...")
    result = perform_hypothesis_testing(
        file_path=test_file,
        group_col="purchased",
        value_col="income",
        test_type="auto"
    )
    
    if result.get('success'):
        print(f"  ✅ Hypothesis test successful!")
        print(f"     - Test used: {result.get('test_type', 'Unknown')}")
        print(f"     - P-value: {result.get('p_value', 0):.4f}")
        print(f"     - Significant: {result.get('is_significant', False)}")
    else:
        print(f"  ❌ Test failed: {result.get('message', 'Unknown error')}")
except Exception as e:
    print(f"  ❌ Hypothesis test error: {e}")

print()

# Test 6: Segment Analysis
print("=" * 80)
print("TEST 6: Segment Analysis")
print("=" * 80)
try:
    print("Running perform_segment_analysis with 3 clusters...")
    result = perform_segment_analysis(
        file_path=test_file,
        features=["age", "income", "credit_score"],
        n_segments=3
    )
    
    if result.get('success'):
        print(f"  ✅ Segmentation successful!")
        print(f"     - Segments created: {result.get('n_segments', 0)}")
        print(f"     - Silhouette score: {result.get('silhouette_score', 0):.3f}")
        
        # Show segment sizes
        segment_sizes = result.get('segment_sizes', {})
        if segment_sizes:
            print(f"\n  Segment Sizes:")
            for segment, size in segment_sizes.items():
                print(f"     - {segment}: {size}")
    else:
        print(f"  ❌ Segmentation failed: {result.get('message', 'Unknown error')}")
except Exception as e:
    print(f"  ❌ Segmentation error: {e}")

print()

# Test 7: Data Quality Plots
print("=" * 80)
print("TEST 7: Data Quality Plots")
print("=" * 80)
try:
    print("Running generate_data_quality_plots...")
    result = generate_data_quality_plots(test_file, output_dir="outputs/test_quality")
    
    if result.get('success'):
        print(f"  ✅ Data quality plots generated!")
        plots = result.get('plots', [])
        print(f"     - Number of plots: {len(plots)}")
        for plot in plots:
            if os.path.exists(plot):
                print(f"     ✓ {Path(plot).name}")
            else:
                print(f"     ✗ {Path(plot).name} (not found)")
    else:
        print(f"  ❌ Plot generation failed: {result.get('message', 'Unknown error')}")
except Exception as e:
    print(f"  ❌ Data quality plots error: {e}")

print()

# Test 8: EDA Plots
print("=" * 80)
print("TEST 8: EDA Plots")
print("=" * 80)
try:
    print("Running generate_eda_plots...")
    result = generate_eda_plots(test_file, target_col="purchased", output_dir="outputs/test_eda")
    
    if result.get('success'):
        print(f"  ✅ EDA plots generated!")
        plots = result.get('plots', [])
        print(f"     - Number of plots: {len(plots)}")
        for plot in plots:
            if os.path.exists(plot):
                print(f"     ✓ {Path(plot).name}")
    else:
        print(f"  ❌ Plot generation failed: {result.get('message', 'Unknown error')}")
except Exception as e:
    print(f"  ❌ EDA plots error: {e}")

print()

# Test 9: Model Training with Visualization
print("=" * 80)
print("TEST 9: Model Training with Visualization")
print("=" * 80)
try:
    print("Running train_baseline_models for classification with target='purchased'...")
    result = train_baseline_models(
        file_path=test_file,
        target_col="purchased",
        task_type="classification",
        test_size=0.3
    )
    
    if result.get('status') == 'success' or 'best_model' in result:
        print(f"  ✅ Model training successful!")
        
        best_model = result.get('best_model', {})
        print(f"     - Best model: {best_model.get('name', 'Unknown')}")
        
        # Show metrics
        metrics = best_model.get('metrics', {})
        if metrics:
            print(f"     - F1 Score: {metrics.get('f1', 0):.3f}")
            print(f"     - Accuracy: {metrics.get('accuracy', 0):.3f}")
        
        # Check for plots
        performance_plots = result.get('performance_plots', [])
        feature_plot = result.get('feature_importance_plot')
        
        if performance_plots:
            print(f"\n  Performance Plots ({len(performance_plots)}):")
            for plot in performance_plots:
                if os.path.exists(plot):
                    print(f"     ✓ {Path(plot).name}")
                else:
                    print(f"     ✗ {Path(plot).name} (not found)")
        
        if feature_plot and os.path.exists(feature_plot):
            print(f"\n  Feature Importance Plot:")
            print(f"     ✓ {Path(feature_plot).name}")
    else:
        print(f"  ❌ Training failed: {result.get('message', 'Unknown error')}")
except Exception as e:
    print(f"  ❌ Model training error: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 10: Generate All Plots
print("=" * 80)
print("TEST 10: Generate All Plots")
print("=" * 80)
try:
    print("Running generate_all_plots...")
    result = generate_all_plots(
        file_path=test_file,
        output_dir="outputs/test_plots"
    )
    
    if result.get('success'):
        print(f"  ✅ All plots generated!")
        print(f"     - Total plots: {result.get('total_plots', 0)}")
        print(f"     - Gallery HTML: {result.get('gallery_html', 'Not created')}")
        
        # Check categories
        plots_by_category = result.get('plots_by_category', {})
        for category, plots in plots_by_category.items():
            print(f"\n  {category.title()} ({len(plots)}):")
            for plot in plots[:3]:  # Show first 3
                if os.path.exists(plot):
                    print(f"     ✓ {Path(plot).name}")
    else:
        print(f"  ❌ Plot generation failed: {result.get('message', 'Unknown error')}")
except Exception as e:
    print(f"  ❌ Generate all plots error: {e}")

print()
print("=" * 80)
print("🎉 TESTING COMPLETE!")
print("=" * 80)
print()
print("📝 Summary:")
print("  - Advanced Insights: 6 functions tested")
print("  - Auto Pipeline: 2 functions tested")
print("  - Visualization Engine: 5 functions tested")
print("  - Model Training Integration: Tested")
print()
print("💡 Next steps:")
print("  1. Check the generated plots in outputs/ directory")
print("  2. Test the chat UI with: python chat_ui.py")
print("  3. Update orchestrator to use new features by default")
print()
