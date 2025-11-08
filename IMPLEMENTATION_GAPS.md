# Implementation Gaps Analysis - What We're Missing

**Date:** November 7, 2025  
**Compared Against:** ai-data-science-team-master repository by business-science  
**Our System:** SDK-based Data Science Copilot with 63 tools

---

## Executive Summary

After implementing **SBERT-based intent classification** from the GitHub repo analysis, we still have **critical functional gaps** that need to be addressed:

### ✅ Already Implemented:
- ✅ 63 comprehensive tools across 10 categories
- ✅ SBERT intent classification (92%+ accuracy)
- ✅ Session management architecture
- ✅ Visualization engine (matplotlib + HTML)
- ✅ Advanced ML features (XGBoost, LightGBM, CatBoost)
- ✅ Production MLOps tools (drift detection, explainability)

### ❌ Critical Missing Features (HIGH PRIORITY):

1. **❌ Smart Data Summarization** - Missing `missing %` and `unique counts` in profile
2. **❌ Multi-Dataset Operations** - No merge/join/concat capabilities
3. **❌ Plotly Interactive Visualizations** - Only static matplotlib plots
4. **❌ Enhanced Data Cleaning** - No smart threshold-based column dropping (>40% missing)
5. **❌ SQL Database Support** - Cannot connect to databases, only CSV/Parquet
6. **❌ EDA Report Generation** - No Sweetviz/DTale/pandas-profiling integration
7. **❌ Session-Based Memory** - Intent classifier exists but no conversation memory

---

## 1. Smart Data Summarization ⚠️ **CRITICAL**

### What They Have (Better):
```python
# From tools/dataframe.py
def get_dataframe_summary():
    # Returns:
    # - Shape (rows × columns)
    # - Column data types
    # - Missing value PERCENTAGE by column  ← WE DON'T HAVE THIS
    # - Unique value COUNTS per column      ← WE DON'T HAVE THIS
    # - First N rows
    # - Descriptive statistics
    # - Safe dict handling (converts to strings)
```

### What We Have (Limited):
```python
# From src/tools/data_profiling.py
def profile_dataset():
    # Returns:
    # - Shape (rows × columns)
    # - Column data types
    # - TOTAL missing values (not per column %)  ← LESS USEFUL
    # - No unique counts per column              ← MISSING
    # - Basic statistics
    # - No dict handling (might crash on dict columns)
```

### The Problem:
```python
# User asks: "Which columns have most missing data?"
# Our system: "Total missing: 1,234" ❌ NOT HELPFUL
# Their system: "Column A: 45%, Column B: 12%, Column C: 3%" ✅ ACTIONABLE
```

### Why It Matters:
- LLM can't make smart cleaning decisions without per-column missing %
- Can't identify high-cardinality categoricals without unique counts
- Dict columns (JSON data) cause errors

### Fix Required:
```python
# src/tools/data_profiling.py - Enhance profile_dataset()

def profile_dataset(file_path: str) -> dict:
    df = pl.read_csv(file_path)
    
    # ADD: Per-column missing percentages
    missing_pct = {}
    for col in df.columns:
        null_count = df[col].null_count()
        missing_pct[col] = {
            "count": null_count,
            "percentage": round((null_count / len(df)) * 100, 2)
        }
    
    # ADD: Unique value counts
    unique_counts = {}
    for col in df.columns:
        try:
            unique_counts[col] = df[col].n_unique()
        except:
            # Handle unhashable types (dicts, lists)
            unique_counts[col] = "N/A (unhashable type)"
    
    return {
        # ... existing fields ...
        "missing_values_per_column": missing_pct,  # NEW
        "unique_counts_per_column": unique_counts,  # NEW
    }
```

**Priority:** 🔴 **HIGH** - Blocks intelligent data cleaning

---

## 2. Multi-Dataset Operations ⚠️ **CRITICAL**

### What They Have:
```python
# From agents/data_wrangling_agent.py
def invoke_agent(
    data_raw: Union[pd.DataFrame, dict, list],  # ← Multiple datasets
    user_instructions: str
):
    # Can handle:
    # - Single DataFrame
    # - List of DataFrames → merge/join them
    # - Dict of named DataFrames → {"customers": df1, "orders": df2}
    
    # Operations:
    # - pd.merge(left, right, on="customer_id", how="inner")
    # - pd.concat([df1, df2], axis=0)  # Vertical stack
    # - pd.concat([df1, df2], axis=1)  # Horizontal stack
```

### What We Have:
```python
# NOTHING - We can only load ONE file at a time
# User uploads: customers.csv, orders.csv
# Our system: "Sorry, I can only work with one file" ❌
```

### Real-World Use Case (We Can't Handle):
```python
# Customer wants to:
# 1. Load customers.csv (customer_id, name, age, city)
# 2. Load orders.csv (order_id, customer_id, product, amount)
# 3. Merge them on customer_id
# 4. Analyze: "Which city has highest average order value?"

# Their system: ✅ Handles this easily
# Our system: ❌ Can't even merge the files
```

### Fix Required:

**Step 1: Add merge_datasets tool**
```python
# NEW FILE: src/tools/data_wrangling.py

import polars as pl
from typing import Optional, Literal

def merge_datasets(
    left_path: str,
    right_path: str,
    output_path: str,
    how: Literal["inner", "left", "right", "outer", "cross"] = "inner",
    on: Optional[str] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    suffix: str = "_right"
) -> dict:
    """
    Merge two datasets using various join strategies.
    
    Args:
        left_path: Path to left dataset
        right_path: Path to right dataset
        output_path: Where to save merged dataset
        how: Join type (inner/left/right/outer/cross)
        on: Column name to join on (if same in both)
        left_on: Column name in left dataset
        right_on: Column name in right dataset
        suffix: Suffix for duplicate column names
    
    Returns:
        {
            "success": bool,
            "output_path": str,
            "left_rows": int,
            "right_rows": int,
            "result_rows": int,
            "merge_type": str
        }
    
    Example:
        >>> merge_datasets(
        ...     "customers.csv", 
        ...     "orders.csv",
        ...     "merged.csv",
        ...     how="left",
        ...     on="customer_id"
        ... )
    """
    try:
        # Load datasets
        left_df = pl.read_csv(left_path)
        right_df = pl.read_csv(right_path)
        
        # Determine join columns
        if on:
            left_on = on
            right_on = on
        elif not left_on or not right_on:
            return {
                "success": False,
                "error": "Must specify 'on' or both 'left_on' and 'right_on'"
            }
        
        # Perform merge
        merged_df = left_df.join(
            right_df,
            left_on=left_on,
            right_on=right_on,
            how=how,
            suffix=suffix
        )
        
        # Save result
        merged_df.write_csv(output_path)
        
        return {
            "success": True,
            "output_path": output_path,
            "left_rows": len(left_df),
            "right_rows": len(right_df),
            "result_rows": len(merged_df),
            "merge_type": how,
            "join_columns": {"left": left_on, "right": right_on}
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def concat_datasets(
    file_paths: list[str],
    output_path: str,
    axis: Literal["vertical", "horizontal"] = "vertical"
) -> dict:
    """
    Concatenate multiple datasets vertically (stack rows) or horizontally (add columns).
    
    Args:
        file_paths: List of file paths to concatenate
        output_path: Where to save result
        axis: "vertical" (stack rows) or "horizontal" (add columns)
    
    Returns:
        {"success": bool, "output_path": str, "result_rows": int, "result_cols": int}
    """
    try:
        dfs = [pl.read_csv(path) for path in file_paths]
        
        if axis == "vertical":
            # Stack rows (union)
            result = pl.concat(dfs, how="vertical")
        else:
            # Add columns side-by-side
            result = pl.concat(dfs, how="horizontal")
        
        result.write_csv(output_path)
        
        return {
            "success": True,
            "output_path": output_path,
            "result_rows": len(result),
            "result_cols": len(result.columns),
            "input_files": len(file_paths)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

**Step 2: Register tools**
```python
# src/tools/__init__.py - Add imports
from .data_wrangling import (
    merge_datasets,
    concat_datasets
)
```

**Step 3: Add to tools_registry.py**
```python
# Add 2 new tools to TOOLS list
{
    "type": "function",
    "function": {
        "name": "merge_datasets",
        "description": "Merge two datasets using SQL-like join operations (inner/left/right/outer)",
        "parameters": { ... }
    }
},
{
    "type": "function",
    "function": {
        "name": "concat_datasets",
        "description": "Concatenate multiple datasets vertically (stack rows) or horizontally (add columns)",
        "parameters": { ... }
    }
}
```

**Step 4: Update Gradio UI to support multiple files**
```python
# chat_ui.py - Change file uploader
file_upload = gr.File(
    label="📁 Upload Dataset(s) - CSV/Parquet",
    file_types=[".csv", ".parquet"],
    file_count="multiple"  # ← ENABLE MULTIPLE FILES
)
```

**Priority:** 🔴 **HIGH** - Core functionality gap, common real-world use case

---

## 3. Plotly Interactive Visualizations ⚠️ **IMPORTANT**

### What They Have (Better):
```python
# From agents/data_visualization_agent.py
import plotly.express as px
import plotly.io as pio

fig = px.scatter(df, x="age", y="income", color="city", hover_data=["name"])
fig_json = pio.to_json(fig)  # JSON-serializable for web

# Features:
# ✅ Interactive (zoom, pan, hover)
# ✅ JSON output for APIs
# ✅ Works in Jupyter, Streamlit, Gradio, web apps
# ✅ Smart chart selection (<10 unique → categorical, >10 → continuous)
```

### What We Have (Limited):
```python
# From src/tools/visualization_engine.py
import matplotlib.pyplot as plt

plt.scatter(df["age"], df["income"])
plt.savefig("plot.png")  # Static PNG

# Limitations:
# ❌ Static images (no interaction)
# ❌ PNG files (large, not web-friendly)
# ❌ No hover info
# ❌ No zoom/pan
# ❌ Not JSON-serializable
```

### Why Plotly is Better:
```python
# Matplotlib (our current):
# - User sees static PNG
# - Can't zoom into outliers
# - Can't hover to see exact values
# - Can't toggle legend to hide series

# Plotly (their approach):
# - User can zoom into interesting regions
# - Hover shows exact data point values
# - Click legend to hide/show series
# - Export to PNG/SVG/HTML
# - JSON format → can send over API
```

### Fix Required:

**Add Plotly dependency**
```bash
# requirements.txt
plotly>=5.18.0
kaleido>=0.2.1  # For static image export
```

**Option 1: Add Plotly Alongside Matplotlib (Recommended)**
```python
# src/tools/visualization_engine.py - ADD Plotly support

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

def generate_scatter_plot(
    file_path: str,
    x_col: str,
    y_col: str,
    output_dir: str = "./outputs/plots",
    engine: Literal["matplotlib", "plotly"] = "plotly"  # NEW
) -> dict:
    """Generate scatter plot with choice of visualization engine."""
    
    if engine == "plotly":
        df = pl.read_csv(file_path).to_pandas()
        
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col,
            title=f"{y_col} vs {x_col}",
            hover_data=df.columns.tolist()
        )
        
        # Save as interactive HTML
        html_path = f"{output_dir}/scatter_{x_col}_vs_{y_col}.html"
        fig.write_html(html_path)
        
        # Also save JSON for API/web use
        json_path = f"{output_dir}/scatter_{x_col}_vs_{y_col}.json"
        with open(json_path, 'w') as f:
            f.write(pio.to_json(fig))
        
        return {
            "success": True,
            "html_path": html_path,
            "json_path": json_path,
            "engine": "plotly",
            "interactive": True
        }
    
    else:
        # Existing matplotlib code
        # ...
```

**Option 2: Replace Matplotlib Entirely (If preferred)**
```python
# Convert all visualization functions to Plotly
# Benefits: Consistent interactive experience
# Drawbacks: Larger file sizes, requires JavaScript
```

**Priority:** 🟡 **MEDIUM-HIGH** - Improves user experience significantly

---

## 4. Enhanced Data Cleaning with Smart Thresholds

### What They Have (Smarter):
```python
# From agents/data_cleaning_agent.py
# Default recommended steps:
1. Remove columns with >40% missing values  ← SMART HEURISTIC
2. Impute numeric with MEAN
3. Impute categorical with MODE
4. Remove extreme outliers (3× IQR)
5. Remove duplicates
```

### What We Have:
```python
# From src/tools/data_cleaning.py
def clean_missing_values(strategy="drop"):
    # Options: drop, mean, median, mode, forward_fill, backward_fill
    # BUT: No automatic "drop column if >40% missing" logic
```

### Fix Required:
```python
# src/tools/data_cleaning.py - Enhance clean_missing_values

def clean_missing_values(
    file_path: str,
    output_path: str,
    strategy: str = "auto",
    threshold: float = 0.4  # NEW PARAMETER
) -> dict:
    """
    Clean missing values with smart threshold-based column dropping.
    
    New 'auto' strategy:
    1. Drop columns with >threshold missing (default 40%)
    2. Impute numeric columns with mean
    3. Impute categorical columns with mode
    4. Forward-fill for time series
    """
    
    df = pl.read_csv(file_path)
    
    if strategy == "auto":
        # Step 1: Drop high-missing columns
        cols_to_drop = []
        for col in df.columns:
            missing_pct = df[col].null_count() / len(df)
            if missing_pct > threshold:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            df = df.drop(cols_to_drop)
            print(f"🗑️  Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing: {cols_to_drop}")
        
        # Step 2: Impute numeric with mean
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        for col in numeric_cols:
            if df[col].null_count() > 0:
                mean_val = df[col].mean()
                df = df.with_columns(pl.col(col).fill_null(mean_val))
        
        # Step 3: Impute categorical with mode
        categorical_cols = [c for c in df.columns if c not in numeric_cols]
        for col in categorical_cols:
            if df[col].null_count() > 0:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "MISSING"
                df = df.with_columns(pl.col(col).fill_null(mode_val))
        
        df.write_csv(output_path)
        
        return {
            "success": True,
            "output_path": output_path,
            "columns_dropped": cols_to_drop,
            "strategy": "auto",
            "threshold": threshold
        }
    
    # ... existing strategy logic ...
```

**Priority:** 🟡 **MEDIUM** - Nice to have, improves automatic cleaning

---

## 5. SQL Database Support ⚠️ **IMPORTANT**

### What They Have:
```python
# From agents/sql_database_agent.py
import sqlalchemy

class SQLDatabaseAgent:
    def __init__(self, connection_string: str):
        self.engine = sqlalchemy.create_engine(connection_string)
        
    # Features:
    # - Connect to PostgreSQL, MySQL, SQLite, SQL Server
    # - List tables
    # - Get table schemas
    # - Execute SQL queries → returns DataFrame
    # - Natural language → SQL query generation
    # - Smart schema filtering (for databases with 100+ tables)
```

### What We Have:
```python
# NOTHING
# We can only work with CSV/Parquet files
# User: "Connect to my production database and analyze customer churn"
# Us: "Sorry, export to CSV first" ❌
```

### Why This Matters:
```python
# Real-world scenario:
# - Production data is in PostgreSQL/MySQL/SQL Server
# - Exporting to CSV is:
#   ❌ Manual, error-prone
#   ❌ Data can be stale by the time analysis is done
#   ❌ Security risk (CSV files scattered everywhere)
#   ❌ Loses data types (dates become strings, etc.)

# Direct SQL connection:
#   ✅ Real-time data
#   ✅ No manual exports
#   ✅ Secure (credentials managed properly)
#   ✅ Preserves data types
```

### Fix Required:

**Step 1: Add dependencies**
```bash
# requirements.txt
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0  # PostgreSQL
pymysql>=1.1.0  # MySQL
pyodbc>=5.0.0  # SQL Server
```

**Step 2: Create SQL tools**
```python
# NEW FILE: src/tools/sql_operations.py

import sqlalchemy as sa
import polars as pl
from typing import Optional

# Global connection pool
_connections = {}

def connect_sql_database(
    connection_string: str,
    connection_id: str = "default",
    test_connection: bool = True
) -> dict:
    """
    Connect to SQL database.
    
    Args:
        connection_string: SQLAlchemy connection string
            Examples:
            - SQLite: "sqlite:///data.db"
            - PostgreSQL: "postgresql://user:password@localhost:5432/dbname"
            - MySQL: "mysql+pymysql://user:password@localhost:3306/dbname"
            - SQL Server: "mssql+pyodbc://user:password@localhost/dbname?driver=ODBC+Driver+17+for+SQL+Server"
        connection_id: Unique ID for this connection
        test_connection: Test connection before returning
    
    Returns:
        {"success": bool, "connection_id": str, "database_type": str}
    """
    try:
        engine = sa.create_engine(connection_string)
        
        if test_connection:
            # Test connection
            with engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
        
        _connections[connection_id] = engine
        
        return {
            "success": True,
            "connection_id": connection_id,
            "database_type": engine.dialect.name,
            "message": f"Connected to {engine.dialect.name} database"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def list_sql_tables(connection_id: str = "default") -> dict:
    """List all tables in connected database."""
    try:
        engine = _connections[connection_id]
        inspector = sa.inspect(engine)
        tables = inspector.get_table_names()
        
        return {
            "success": True,
            "tables": tables,
            "count": len(tables)
        }
    
    except KeyError:
        return {"success": False, "error": f"No connection with ID '{connection_id}'"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def query_sql_database(
    query: str,
    connection_id: str = "default",
    output_path: Optional[str] = None
) -> dict:
    """
    Execute SQL query and return results.
    
    Args:
        query: SQL query to execute
        connection_id: Which connection to use
        output_path: Optional path to save results as CSV
    
    Returns:
        {"success": bool, "rows": int, "columns": list, "output_path": str}
    """
    try:
        engine = _connections[connection_id]
        
        # Execute query using Polars (faster than pandas)
        df = pl.read_database(query, engine)
        
        # Save if output path provided
        if output_path:
            df.write_csv(output_path)
        
        return {
            "success": True,
            "rows": len(df),
            "columns": df.columns,
            "output_path": output_path,
            "preview": df.head(10).to_dicts()  # First 10 rows
        }
    
    except KeyError:
        return {"success": False, "error": f"No connection with ID '{connection_id}'"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_table_schema(
    table_name: str,
    connection_id: str = "default"
) -> dict:
    """Get schema information for a table."""
    try:
        engine = _connections[connection_id]
        inspector = sa.inspect(engine)
        
        columns = inspector.get_columns(table_name)
        primary_keys = inspector.get_pk_constraint(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        
        return {
            "success": True,
            "table_name": table_name,
            "columns": [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col["nullable"]
                }
                for col in columns
            ],
            "primary_keys": primary_keys.get("constrained_columns", []),
            "foreign_keys": [
                {
                    "column": fk["constrained_columns"][0],
                    "references": f"{fk['referred_table']}.{fk['referred_columns'][0]}"
                }
                for fk in foreign_keys
            ]
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Step 3: Register SQL tools in tools_registry.py**

**Priority:** 🟡 **MEDIUM-HIGH** - Expands use cases significantly, production environments

---

## 6. EDA Report Generation

### What They Have:
```python
# From tools/eda.py
import sweetviz as sv
import dtale
import ydata_profiling

# Full HTML interactive reports:
sv.analyze(df).show_html("report.html")
```

### What We Have:
- Only basic profiling statistics
- No comprehensive HTML reports

### Fix Required:
```bash
# requirements.txt
sweetviz>=2.3.1
ydata-profiling>=4.6.0
dtale>=3.10.0
```

```python
# NEW: src/tools/eda_reports.py

def generate_eda_report(
    file_path: str,
    output_dir: str = "./outputs/reports",
    report_type: Literal["sweetviz", "ydata", "dtale"] = "sweetviz"
) -> dict:
    """Generate comprehensive EDA report."""
    # Implementation...
```

**Priority:** 🟢 **LOW-MEDIUM** - Nice to have, but we have basic profiling

---

## 7. Session-Based Conversational Memory ⚠️ **CRITICAL**

### Current Status:
- ✅ SBERT intent classifier implemented
- ❌ No session memory (each query is independent)
- ❌ Can't reference "it", "that", "the previous result"

### What's Missing:
```python
# User: "Train a model on customer_churn.csv"
# Agent: [Trains model, best model: XGBoost with 0.85 F1 score]

# User: "Show me a confusion matrix for it"  ← "it" = XGBoost model
# Our Agent: ❌ "What is 'it'? Please specify."
# Their Agent: ✅ "Here's the confusion matrix for XGBoost"
```

### Fix Required (Already Documented in GITHUB_REPO_ANALYSIS.md Section 11):

Implement `SessionManager` class from the analysis document.

**Priority:** 🔴 **HIGH** - Makes the agent feel conversational vs robotic

---

## Priority Ranking for Implementation

### 🔴 Phase 1: Critical Gaps (Week 1-2)
1. **Session Memory** - Makes agent conversational
2. **Smart Data Summarization** - Better LLM decision-making
3. **Multi-Dataset Operations** - Core real-world functionality

### 🟡 Phase 2: Important Enhancements (Week 3-4)
4. **Plotly Visualizations** - Better UX
5. **Enhanced Data Cleaning** - Smarter automation
6. **SQL Database Support** - Production use cases

### 🟢 Phase 3: Nice-to-Have (Week 5-6)
7. **EDA Report Generation** - Comprehensive insights
8. **Smart Schema Filtering** - Large database support

---

## Implementation Checklist

### Phase 1 (Critical - Complete First)
- [ ] Implement `SessionManager` class for conversation memory
- [ ] Add per-column missing % and unique counts to `profile_dataset()`
- [ ] Create `merge_datasets()` and `concat_datasets()` tools
- [ ] Update Gradio UI to support multiple file uploads
- [ ] Test multi-file workflow: upload 2 files → merge → analyze

### Phase 2 (Important - Do Next)
- [ ] Add Plotly dependency to requirements.txt
- [ ] Create Plotly versions of scatter, line, bar, histogram plots
- [ ] Update `generate_all_plots()` to support engine="plotly"
- [ ] Enhance `clean_missing_values()` with smart threshold dropping
- [ ] Add SQL database connection tools (`connect_sql_database`, etc.)
- [ ] Test SQL workflow: connect → query → analyze

### Phase 3 (Nice-to-Have - If Time Permits)
- [ ] Add Sweetviz/ydata-profiling for EDA reports
- [ ] Implement smart schema filtering for large databases
- [ ] Add export-to-dashboard feature (Streamlit/Dash)

---

## Key Takeaway

**We have more tools (63) than they do (~15), BUT:**
- ❌ We lack **multi-dataset operations** (merge/join)
- ❌ We lack **better data summarization** (per-column missing %)
- ❌ We lack **interactive visualizations** (Plotly)
- ❌ We lack **conversation memory** (session management)
- ❌ We lack **SQL database support**

**Their system is better for:**
- Real-world multi-file workflows
- Production database analysis
- Interactive exploration

**Our system is better for:**
- Advanced ML (63 specialized tools)
- Production MLOps (drift detection, explainability)
- Automation (auto_ml_pipeline, auto_feature_selection)

**Recommendation:** Implement Phase 1 features to match their core capabilities while keeping our advanced tool advantage.
