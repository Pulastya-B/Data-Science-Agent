"""
Data Cleaning Tools
Tools for handling missing values, outliers, and data type issues.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.polars_helpers import (
    load_dataframe,
    save_dataframe,
    get_numeric_columns,
    get_categorical_columns,
    get_datetime_columns,
    detect_id_columns,
)
from utils.validation import (
    validate_file_exists,
    validate_file_format,
    validate_dataframe,
    validate_columns_exist,
)


def clean_missing_values(file_path: str, strategy: Dict[str, str], 
                        output_path: str) -> Dict[str, Any]:
    """
    Handle missing values using appropriate strategies.
    
    Args:
        file_path: Path to CSV or Parquet file
        strategy: Dictionary mapping column names to strategies 
                 ('median', 'mean', 'mode', 'forward_fill', 'drop', 'auto')
        output_path: Path to save cleaned dataset
        
    Returns:
        Dictionary with cleaning report
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    # Get column type information
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    id_cols = detect_id_columns(df)
    
    report = {
        "original_rows": len(df),
        "columns_processed": {},
        "rows_dropped": 0
    }
    
    # Process each column based on strategy
    for col, strat in strategy.items():
        if col not in df.columns:
            report["columns_processed"][col] = {
                "status": "error",
                "message": f"Column not found"
            }
            continue
        
        null_count_before = df[col].null_count()
        
        if null_count_before == 0:
            report["columns_processed"][col] = {
                "status": "skipped",
                "message": "No missing values"
            }
            continue
        
        # Don't impute ID columns
        if col in id_cols and strat != "drop":
            report["columns_processed"][col] = {
                "status": "skipped",
                "message": "ID column - not imputed"
            }
            continue
        
        # Auto strategy: choose based on column type
        if strat == "auto":
            if col in numeric_cols:
                strat = "median"
            elif col in categorical_cols:
                strat = "mode"
            else:
                strat = "drop"
        
        # Apply strategy
        try:
            if strat == "median":
                if col in numeric_cols:
                    median_val = df[col].median()
                    df = df.with_columns(
                        pl.col(col).fill_null(median_val).alias(col)
                    )
                else:
                    report["columns_processed"][col] = {
                        "status": "error",
                        "message": "Cannot use median on non-numeric column"
                    }
                    continue
            
            elif strat == "mean":
                if col in numeric_cols:
                    mean_val = df[col].mean()
                    df = df.with_columns(
                        pl.col(col).fill_null(mean_val).alias(col)
                    )
                else:
                    report["columns_processed"][col] = {
                        "status": "error",
                        "message": "Cannot use mean on non-numeric column"
                    }
                    continue
            
            elif strat == "mode":
                mode_val = df[col].drop_nulls().mode().first()
                if mode_val is not None:
                    df = df.with_columns(
                        pl.col(col).fill_null(mode_val).alias(col)
                    )
            
            elif strat == "forward_fill":
                df = df.with_columns(
                    pl.col(col).forward_fill().alias(col)
                )
            
            elif strat == "drop":
                df = df.filter(pl.col(col).is_not_null())
            
            else:
                report["columns_processed"][col] = {
                    "status": "error",
                    "message": f"Unknown strategy: {strat}"
                }
                continue
            
            null_count_after = df[col].null_count()
            
            report["columns_processed"][col] = {
                "status": "success",
                "strategy": strat,
                "nulls_before": int(null_count_before),
                "nulls_after": int(null_count_after),
                "nulls_handled": int(null_count_before - null_count_after)
            }
        
        except Exception as e:
            report["columns_processed"][col] = {
                "status": "error",
                "message": str(e)
            }
    
    report["final_rows"] = len(df)
    report["rows_dropped"] = report["original_rows"] - report["final_rows"]
    
    # Save cleaned dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_dataframe(df, output_path)
    report["output_path"] = output_path
    
    return report


def handle_outliers(file_path: str, method: str, columns: List[str], 
                   output_path: str) -> Dict[str, Any]:
    """
    Detect and handle outliers in numeric columns.
    
    Args:
        file_path: Path to CSV or Parquet file
        method: Method to handle outliers ('clip', 'winsorize', 'remove')
        columns: List of columns to check, or ['all'] for all numeric columns
        output_path: Path to save cleaned dataset
        
    Returns:
        Dictionary with outlier handling report
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    # Determine which columns to process
    numeric_cols = get_numeric_columns(df)
    
    if columns == ["all"]:
        target_cols = numeric_cols
    else:
        # Validate columns exist and are numeric
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found")
            if col not in numeric_cols:
                raise ValueError(f"Column '{col}' is not numeric")
        target_cols = columns
    
    report = {
        "original_rows": len(df),
        "method": method,
        "columns_processed": {}
    }
    
    # Process each column
    for col in target_cols:
        col_data = df[col].drop_nulls()
        
        if len(col_data) == 0:
            report["columns_processed"][col] = {
                "status": "skipped",
                "message": "All values are null"
            }
            continue
        
        # Calculate IQR bounds
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Count outliers
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outliers_mask.sum()
        
        if outlier_count == 0:
            report["columns_processed"][col] = {
                "status": "skipped",
                "message": "No outliers detected"
            }
            continue
        
        # Apply method
        if method == "clip":
            # Clip values to bounds
            df = df.with_columns(
                pl.col(col).clip(lower_bound, upper_bound).alias(col)
            )
        
        elif method == "winsorize":
            # Winsorize: cap at 1st and 99th percentiles
            p1 = col_data.quantile(0.01)
            p99 = col_data.quantile(0.99)
            df = df.with_columns(
                pl.col(col).clip(p1, p99).alias(col)
            )
        
        elif method == "remove":
            # Remove rows with outliers
            df = df.filter(~outliers_mask)
        
        report["columns_processed"][col] = {
            "status": "success",
            "outliers_detected": int(outlier_count),
            "bounds": {
                "lower": float(lower_bound),
                "upper": float(upper_bound)
            }
        }
    
    report["final_rows"] = len(df)
    report["rows_dropped"] = report["original_rows"] - report["final_rows"]
    
    # Save cleaned dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_dataframe(df, output_path)
    report["output_path"] = output_path
    
    return report


def fix_data_types(file_path: str, type_mapping: Optional[Dict[str, str]] = None,
                  output_path: str = None) -> Dict[str, Any]:
    """
    Auto-detect and fix incorrect data types.
    
    Args:
        file_path: Path to CSV or Parquet file
        type_mapping: Optional dictionary mapping columns to target types
                     ('int', 'float', 'string', 'date', 'bool', 'category')
                     Use 'auto' or None for automatic detection
        output_path: Path to save dataset with fixed types
        
    Returns:
        Dictionary with type fixing report
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    if type_mapping is None or type_mapping == {"auto": "auto"}:
        type_mapping = {}
    
    report = {
        "columns_processed": {}
    }
    
    for col in df.columns:
        original_dtype = str(df[col].dtype)
        
        # Get target type from mapping or auto-detect
        if col in type_mapping and type_mapping[col] != "auto":
            target_type = type_mapping[col]
        else:
            # Auto-detect target type
            target_type = _auto_detect_type(df[col])
        
        if target_type is None:
            report["columns_processed"][col] = {
                "status": "skipped",
                "original_dtype": original_dtype,
                "message": "Could not auto-detect type"
            }
            continue
        
        # Try to convert
        try:
            if target_type == "int":
                df = df.with_columns(
                    pl.col(col).cast(pl.Int64, strict=False).alias(col)
                )
            elif target_type == "float":
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False).alias(col)
                )
            elif target_type == "string":
                df = df.with_columns(
                    pl.col(col).cast(pl.Utf8).alias(col)
                )
            elif target_type == "date":
                df = df.with_columns(
                    pl.col(col).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias(col)
                )
            elif target_type == "bool":
                df = df.with_columns(
                    pl.col(col).cast(pl.Boolean, strict=False).alias(col)
                )
            elif target_type == "category":
                df = df.with_columns(
                    pl.col(col).cast(pl.Categorical).alias(col)
                )
            
            new_dtype = str(df[col].dtype)
            
            report["columns_processed"][col] = {
                "status": "success",
                "original_dtype": original_dtype,
                "new_dtype": new_dtype,
                "target_type": target_type
            }
        
        except Exception as e:
            report["columns_processed"][col] = {
                "status": "error",
                "original_dtype": original_dtype,
                "target_type": target_type,
                "message": str(e)
            }
    
    # Save dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_dataframe(df, output_path)
    report["output_path"] = output_path
    
    return report


def _auto_detect_type(series: pl.Series) -> Optional[str]:
    """
    Auto-detect appropriate type for a series.
    
    Args:
        series: Polars series
        
    Returns:
        Detected type string or None
    """
    # Already correct type
    if series.dtype in pl.NUMERIC_DTYPES:
        return None
    
    if series.dtype in [pl.Date, pl.Datetime]:
        return None
    
    # Try to detect from string values
    if series.dtype == pl.Utf8:
        sample = series.drop_nulls().head(100)
        
        if len(sample) == 0:
            return None
        
        # Check for boolean
        unique_vals = set(str(v).lower() for v in sample.to_list())
        if unique_vals.issubset({'true', 'false', '1', '0', 'yes', 'no', 't', 'f'}):
            return "bool"
        
        # Check for numeric
        try:
            sample.cast(pl.Float64)
            # Check if all are integers
            if all('.' not in str(v) for v in sample.to_list() if v is not None):
                return "int"
            return "float"
        except:
            pass
        
        # Check for date
        try:
            sample.str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            return "date"
        except:
            pass
        
        # Check if should be categorical (low cardinality)
        n_unique = series.n_unique()
        if n_unique < len(series) * 0.5 and n_unique < 100:
            return "category"
    
    return None
