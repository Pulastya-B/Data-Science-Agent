"""Tools module initialization - All 44 tools."""

# Basic Tools (10)
from .data_profiling import (
    profile_dataset,
    detect_data_quality_issues,
    analyze_correlations
)

from .data_cleaning import (
    clean_missing_values,
    handle_outliers,
    fix_data_types
)

from .data_type_conversion import (
    force_numeric_conversion,
    smart_type_inference
)

from .feature_engineering import (
    create_time_features,
    encode_categorical
)

from .model_training import (
    train_baseline_models,
    generate_model_report
)

# Advanced Analysis Tools (5)
from .advanced_analysis import (
    perform_eda_analysis,
    detect_model_issues,
    detect_anomalies,
    detect_and_handle_multicollinearity,
    perform_statistical_tests
)

# Advanced Feature Engineering Tools (4)
from .advanced_feature_engineering import (
    create_interaction_features,
    create_aggregation_features,
    engineer_text_features,
    auto_feature_engineering
)

# Advanced Preprocessing Tools (3)
from .advanced_preprocessing import (
    handle_imbalanced_data,
    perform_feature_scaling,
    split_data_strategically
)

# Advanced Training Tools (3)
from .advanced_training import (
    hyperparameter_tuning,
    train_ensemble_models,
    perform_cross_validation
)

# Business Intelligence Tools (4)
from .business_intelligence import (
    perform_cohort_analysis,
    perform_rfm_analysis,
    detect_causal_relationships,
    generate_business_insights
)

# Computer Vision Tools (3)
from .computer_vision import (
    extract_image_features,
    perform_image_clustering,
    analyze_tabular_image_hybrid
)

# NLP/Text Analytics Tools (4)
from .nlp_text_analytics import (
    perform_topic_modeling,
    perform_named_entity_recognition,
    analyze_sentiment_advanced,
    perform_text_similarity
)

# Production/MLOps Tools (5)
from .production_mlops import (
    monitor_model_drift,
    explain_predictions,
    generate_model_card,
    perform_ab_test_analysis,
    detect_feature_leakage
)

# Time Series Tools (3)
from .time_series import (
    forecast_time_series,
    detect_seasonality_trends,
    create_time_series_features
)

from .tools_registry import TOOLS, get_tool_by_name, get_all_tool_names

__all__ = [
    # Basic Data Profiling (3)
    "profile_dataset",
    "detect_data_quality_issues",
    "analyze_correlations",
    
    # Basic Data Cleaning (3)
    "clean_missing_values",
    "handle_outliers",
    "fix_data_types",
    
    # Data Type Conversion (2)
    "force_numeric_conversion",
    "smart_type_inference",
    
    # Basic Feature Engineering (2)
    "create_time_features",
    "encode_categorical",
    
    # Basic Model Training (2)
    "train_baseline_models",
    "generate_model_report",
    
    # Advanced Analysis (5)
    "perform_eda_analysis",
    "detect_model_issues",
    "detect_anomalies",
    "detect_and_handle_multicollinearity",
    "perform_statistical_tests",
    
    # Advanced Feature Engineering (4)
    "create_interaction_features",
    "create_aggregation_features",
    "engineer_text_features",
    "auto_feature_engineering",
    
    # Advanced Preprocessing (3)
    "handle_imbalanced_data",
    "perform_feature_scaling",
    "split_data_strategically",
    
    # Advanced Training (3)
    "hyperparameter_tuning",
    "train_ensemble_models",
    "perform_cross_validation",
    
    # Business Intelligence (4)
    "perform_cohort_analysis",
    "perform_rfm_analysis",
    "detect_causal_relationships",
    "generate_business_insights",
    
    # Computer Vision (3)
    "extract_image_features",
    "perform_image_clustering",
    "analyze_tabular_image_hybrid",
    
    # NLP/Text Analytics (4)
    "perform_topic_modeling",
    "perform_named_entity_recognition",
    "analyze_sentiment_advanced",
    "perform_text_similarity",
    
    # Production/MLOps (5)
    "monitor_model_drift",
    "explain_predictions",
    "generate_model_card",
    "perform_ab_test_analysis",
    "detect_feature_leakage",
    
    # Time Series (3)
    "forecast_time_series",
    "detect_seasonality_trends",
    "create_time_series_features",
    
    # Registry
    "TOOLS",
    "get_tool_by_name",
    "get_all_tool_names",
]
