"""Tools module initialization."""

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

from .feature_engineering import (
    create_time_features,
    encode_categorical
)

from .model_training import (
    train_baseline_models,
    generate_model_report
)

from .tools_registry import TOOLS, get_tool_by_name, get_all_tool_names

__all__ = [
    # Data Profiling
    "profile_dataset",
    "detect_data_quality_issues",
    "analyze_correlations",
    
    # Data Cleaning
    "clean_missing_values",
    "handle_outliers",
    "fix_data_types",
    
    # Feature Engineering
    "create_time_features",
    "encode_categorical",
    
    # Model Training
    "train_baseline_models",
    "generate_model_report",
    
    # Registry
    "TOOLS",
    "get_tool_by_name",
    "get_all_tool_names",
]
