# Data Science Agent 🤖

A production-grade **autonomous AI agent** for end-to-end data science workflows. Upload datasets, describe your goals in natural language, and let the AI handle everything from data profiling to model training.

## 🎯 Project Goal

Achieve **50-70th percentile performance** on Kaggle competitions through intelligent automation, proving that AI agents can handle real-world ML tasks with minimal human intervention.

## 🌟 What Makes This Unique

- **75 Specialized Tools**: Complete ML toolkit from profiling to deployment
- **Dual LLM Support**: Groq (llama-3.3-70b) + Google Gemini (2.0-flash-exp)
- **Smart Orchestration**: LLM-driven function calling with intelligent tool chaining
- **Token Optimized**: Compressed tool registry (34% reduction) for large datasets
- **Production Ready**: SQLite caching, error recovery, rate limiting
- **Multiple Interfaces**: CLI, Web UI (Gradio), REST API (FastAPI)
- **Cloud Native**: Ready for Google Cloud Run deployment

## ✨ Features

### 🧠 Intelligent Orchestration
- **Native Function Calling**: Direct integration with Groq/Gemini APIs (no LangChain/CrewAI complexity)
- **Smart Tool Selection**: LLM intelligently routes to appropriate tools
- **Adaptive Workflows**: Automatically adjusts strategy based on data characteristics
- **Error Recovery**: Automatic retry with parameter correction
- **Anti-Repetition System**: Prevents infinite loops with "ONCE ONLY" directives
- **Context-Aware**: Learns from previous steps to optimize next actions

### 🛠️ Comprehensive Tool Suite (75 Tools)

#### 📊 Data Profiling & Analysis (7 tools)
- `profile_dataset`: Complete statistics, types, memory usage
- `detect_data_quality_issues`: Outliers, duplicates, inconsistencies
- `analyze_correlations`: Feature relationships and target correlations
- `get_smart_summary`: Enhanced profiling with missing % and unique counts
- `perform_eda_analysis`: Comprehensive exploratory data analysis
- `detect_anomalies`: Statistical anomaly detection
- `perform_statistical_tests`: Hypothesis testing and significance

#### 🧹 Data Cleaning (8 tools)
- `clean_missing_values`: Smart imputation (auto/median/mean/mode/forward_fill)
- `handle_outliers`: IQR-based detection with clip/winsorize/remove
- `fix_data_types`: Auto-detect and fix incorrect types
- `force_numeric_conversion`: Convert strings to numeric
- `smart_type_inference`: Intelligent type detection
- `remove_duplicates`: Handle duplicate rows
- `filter_rows`: Conditional filtering
- `drop_columns`: Column removal

#### 🔧 Feature Engineering (13 tools)
- `encode_categorical`: One-hot, target, frequency encoding
- `create_time_features`: Extract temporal patterns (cyclical encoding)
- `create_interaction_features`: Feature crosses and products
- `create_aggregation_features`: Rolling windows and aggregations
- `engineer_text_features`: TF-IDF, word counts, sentiment
- `auto_feature_engineering`: Automated feature discovery
- `create_ratio_features`: Ratios between numeric columns
- `create_statistical_features`: Rolling stats (mean, std, min, max)
- `create_log_features`: Log transformations for skewed data
- `create_binned_features`: Discretization and bucketing

#### 🤖 Model Training & Optimization (6 tools)
- `train_baseline_models`: Train LR, RF, XGBoost, LightGBM, CatBoost
- `hyperparameter_tuning`: Optuna-based optimization
- `train_ensemble_models`: Stacking and voting ensembles
- `perform_cross_validation`: K-fold validation strategies
- `auto_ml_pipeline`: Zero-config end-to-end pipeline
- `auto_feature_selection`: SelectKBest, mutual information

#### 📈 Visualization (11 tools)
- **Static (Matplotlib/Seaborn)**: Quality plots, EDA plots, distributions
- **Interactive (Plotly)**: Scatter, histogram, heatmap, box plots, time series
- `generate_plotly_dashboard`: Comprehensive interactive dashboard
- `generate_all_plots`: Complete visualization suite

#### 📑 EDA Reports (3 tools)
- `generate_sweetviz_report`: Beautiful fast HTML reports
- `generate_ydata_profiling_report`: Comprehensive detailed analysis
- `generate_combined_eda_report`: Both Sweetviz + ydata-profiling

#### 🔍 Advanced Analysis (11 tools)
- `detect_model_issues`: Overfitting, underfitting, data leakage
- `detect_and_handle_multicollinearity`: VIF-based detection
- `analyze_root_cause`: Identify key drivers
- `detect_trends_and_seasonality`: Time series patterns
- `perform_hypothesis_testing`: Statistical significance
- `analyze_distribution`: Normality tests and transformations
- `perform_segment_analysis`: Cohort and segment insights

#### 🚀 Production & MLOps (5 tools)
- `monitor_model_drift`: Detect distribution shifts
- `explain_predictions`: SHAP and LIME explanations
- `generate_model_card`: Model documentation
- `perform_ab_test_analysis`: Experiment analysis
- `detect_feature_leakage`: Data leakage detection

#### 📊 Business Intelligence (4 tools)
- `perform_cohort_analysis`: User cohort behavior
- `perform_rfm_analysis`: RFM customer segmentation
- `detect_causal_relationships`: Causal inference
- `generate_business_insights`: Automated insights

#### 🖼️ Computer Vision (3 tools)
- `extract_image_features`: CNN-based feature extraction
- `perform_image_clustering`: K-means on image features
- `analyze_tabular_image_hybrid`: Multi-modal analysis

#### 📝 NLP & Text Analytics (4 tools)
- `perform_topic_modeling`: LDA and BERTopic
- `perform_named_entity_recognition`: NER extraction
- `analyze_sentiment_advanced`: Sentiment analysis
- `perform_text_similarity`: Semantic similarity

#### ⏰ Time Series (3 tools)
- `forecast_time_series`: Prophet-based forecasting
- `detect_seasonality_trends`: Decomposition and patterns
- `create_time_series_features`: Lag, rolling, seasonal features

#### 🔗 Data Wrangling (3 tools)
- `merge_datasets`: SQL-like joins (inner/left/right/outer/cross)
- `concat_datasets`: Vertical and horizontal stacking
- `reshape_dataset`: Pivot and melt operations

### ⚡ Performance Optimization
- **SQLite Caching**: Memoization of expensive operations (24hr TTL)
- **Polars & DuckDB**: 10-100x faster than Pandas
- **Token Optimization**: Compressed schemas (8,193 → 5,463 tokens, 34% reduction)
- **Smart Summarization**: Tool results compressed by 90%
- **Rate Limiting**: Intelligent API management (Gemini: 3.5s, Groq: 0s)
- **Parallel Execution**: Independent tools run concurrently

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Groq API key ([Get one here](https://console.groq.com)) OR Google API key
- 8GB+ RAM recommended for large datasets

### Installation

```bash
# Clone the repository
git clone https://github.com/Surfing-Ninja/Data-Science-Agent.git
cd Data-Science-Agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with:
GROQ_API_KEY=your_groq_key_here
# OR
GOOGLE_API_KEY=your_google_key_here
LLM_PROVIDER=groq  # or 'gemini'
```

### Usage Options

#### 1️⃣ CLI Interface (Command Line)

```bash
# Complete analysis workflow
python src/cli.py analyze data.csv --target Survived --task "Predict survival"

# Quick profile
python src/cli.py profile data.csv

# Train models only
python src/cli.py train cleaned.csv price --task-type regression

# Cache management
python src/cli.py cache-stats
python src/cli.py clear-cache
```

#### 2️⃣ Web UI (Gradio Interface)

```bash
# Start the web interface
python chat_ui.py

# Open browser to http://localhost:7860
# Upload CSV/Parquet files
# Chat: "Analyze this dataset and predict house prices"
# Use quick actions: Profile, Train, Clean
```

#### 3️⃣ REST API (FastAPI Server)

```bash
# Start API server
cd src/api
python app.py

# Server runs on http://localhost:8080

# Example: Full analysis
curl -X POST http://localhost:8080/run \
  -F "file=@data.csv" \
  -F "task_description=Predict passenger survival" \
  -F "target_col=Survived"

# Example: Quick profile
curl -X POST http://localhost:8080/profile \
  -F "file=@data.csv"

# List available tools
curl http://localhost:8080/tools
```

#### 4️⃣ Python API (Programmatic)

```python
from src.orchestrator import DataScienceCopilot

# Initialize agent
agent = DataScienceCopilot(
    reasoning_effort="medium",  # low/medium/high
    provider="groq"  # or 'gemini'
)

# Run complete workflow
result = agent.analyze(
    file_path="titanic.csv",
    task_description="Predict passenger survival with feature engineering",
    target_col="Survived",
    use_cache=True,
    max_iterations=20
)

# Access results
print(result["status"])  # success/error
print(result["summary"])  # Final summary
print(result["workflow_history"])  # All steps taken
print(result["execution_time"])  # Time taken
print(result["api_calls_made"])  # LLM calls
```

### Example Workflow

See `examples/titanic_example.py` for a complete end-to-end example:

```python
from src.orchestrator import DataScienceCopilot
from rich.console import Console

console = Console()
agent = DataScienceCopilot(reasoning_effort="medium")

# Define task
task = """
Analyze the Titanic dataset and build a survival prediction model.
Steps:
1. Understand data structure and quality issues
2. Handle missing values appropriately  
3. Engineer relevant features (family size, titles from names)
4. Train and compare baseline models
5. Identify most important features
"""

# Run analysis
result = agent.analyze(
    file_path="./data/titanic.csv",
    task_description=task,
    target_col="Survived",
    use_cache=True
)

# Display results
if result["status"] == "success":
    console.print("✅ Analysis Complete!")
    console.print(result["summary"])
```

## 🏗️ Architecture

### Design Philosophy
- **No Heavy Frameworks**: Pure LLM SDK function calling (NO LangChain/CrewAI/LangGraph)
- **Single Orchestrator**: One intelligent router instead of multi-agent complexity
- **Fast Iteration**: Optimize for debugging and rapid development
- **Actionable Results**: Every tool returns structured, LLM-parseable outputs
- **Modular Tools**: Each tool is independent and composable
- **Cloud Ready**: Adapters for local and cloud execution

### System Flow

```
User Query (Natural Language)
    ↓
DataScienceCopilot Orchestrator
    ↓
LLM Function Calling (Groq/Gemini)
    ↓
Tool Selection & Parameter Extraction
    ↓
Tool Execution (Parallel where possible)
    ↓
Result Summarization (90% compression)
    ↓
Next Action Decision
    ↓
Final Report Generation
```

### 8-Step ML Pipeline

```
1. profile_dataset()              → Dataset statistics
2. detect_data_quality_issues()   → Find outliers, duplicates
3. generate_data_quality_plots()  → Visualize issues
4. clean_missing_values()         → Smart imputation
5. handle_outliers()              → IQR-based clipping
6. force_numeric_conversion()     → String→numeric
7. encode_categorical()           → One-hot/target encoding
8. generate_eda_plots()           → Visualizations
9. [Optional] train_baseline_models()  → If ML requested
10. [Optional] generate_combined_eda_report()  → HTML reports
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Groq (llama-3.3-70b) / Gemini (2.0-flash) | Function calling & reasoning |
| **Data Processing** | Polars | 10-100x faster dataframes |
| **SQL** | DuckDB | Complex queries |
| **ML Core** | scikit-learn | Base algorithms |
| **Gradient Boosting** | XGBoost, LightGBM, CatBoost | Advanced models |
| **Hyperparameter Tuning** | Optuna | Bayesian optimization |
| **Explainability** | SHAP, LIME | Model interpretation |
| **Imbalanced Data** | imbalanced-learn | SMOTE, sampling |
| **Visualization** | Matplotlib, Seaborn, Plotly | Static & interactive plots |
| **EDA Reports** | Sweetviz, ydata-profiling | Automated HTML reports |
| **Caching** | SQLite | Operation memoization |
| **CLI** | Typer + Rich | Beautiful terminal UI |
| **Web UI** | Gradio | Interactive interface |
| **API** | FastAPI | REST endpoints |
| **Time Series** | Prophet | Forecasting |
| **NLP** | TextBlob, spaCy (optional) | Text analytics |

### Token Optimization Strategy

**Problem**: Hit Groq's 12K tokens/minute limit with large datasets

**Solution**:
1. **Compressed Tool Registry** (34% reduction)
   - Before: 8,193 tokens
   - After: 5,463 tokens
   - Method: Truncate descriptions, remove verbose parameter docs
   
2. **Smart Result Summarization** (90% reduction)
   - Profile results: 5,000 → 200 tokens
   - Only send essential metrics to LLM
   
3. **Conversation History Pruning**
   - Keep only last 8 messages
   - Prevents unbounded memory growth
   
4. **Result**: ~6,000 tokens available for tool results (50% of limit)

## 📁 Project Structure

```
Data-Science-Agent/
├── src/
│   ├── orchestrator.py          # Main DataScienceCopilot class (1,136 lines)
│   ├── cli.py                   # Typer CLI interface
│   ├── api/
│   │   └── app.py              # FastAPI server for Cloud Run
│   ├── tools/                   # 75 specialized tools
│   │   ├── data_profiling.py
│   │   ├── data_cleaning.py
│   │   ├── data_type_conversion.py
│   │   ├── feature_engineering.py
│   │   ├── advanced_feature_engineering.py
│   │   ├── enhanced_feature_engineering.py
│   │   ├── model_training.py
│   │   ├── advanced_training.py
│   │   ├── visualization_engine.py
│   │   ├── matplotlib_visualizations.py
│   │   ├── plotly_visualizations.py
│   │   ├── eda_reports.py
│   │   ├── advanced_analysis.py
│   │   ├── advanced_insights.py
│   │   ├── advanced_preprocessing.py
│   │   ├── business_intelligence.py
│   │   ├── computer_vision.py
│   │   ├── nlp_text_analytics.py
│   │   ├── time_series.py
│   │   ├── production_mlops.py
│   │   ├── auto_pipeline.py
│   │   ├── data_wrangling.py
│   │   └── tools_registry.py   # All tool definitions (1,514 lines)
│   ├── cache/
│   │   └── cache_manager.py    # SQLite caching system
│   └── utils/
│       ├── polars_helpers.py   # Data manipulation utilities
│       └── validation.py       # Input validation
├── examples/
│   └── titanic_example.py      # Complete workflow demo
├── data/                        # Sample datasets
├── outputs/                     # Generated outputs
│   ├── data/                   # Cleaned/processed datasets
│   ├── models/                 # Trained model files (.pkl)
│   ├── plots/                  # Visualizations (.png, .html)
│   │   ├── quality/
│   │   ├── eda/
│   │   ├── interactive/
│   │   └── model_performance/
│   └── reports/                # EDA reports (.html)
├── temp/                        # Temporary file storage
├── cache_db/                    # SQLite cache database
├── chat_ui.py                   # Gradio web interface (912 lines)
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (API keys)
└── README.md                    # This file
```

## 🔧 Configuration

### Environment Variables (.env file)

```env
# ============================================
# LLM Provider Configuration
# ============================================
LLM_PROVIDER=groq              # 'groq' or 'gemini'

# Groq API (llama-3.3-70b-versatile)
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile
REASONING_EFFORT=medium        # low, medium, high

# Google Gemini API (gemini-2.0-flash-exp)
GOOGLE_API_KEY=your_google_key_here
GEMINI_MODEL=gemini-2.0-flash-exp

# ============================================
# Performance & Caching
# ============================================
CACHE_DB_PATH=./cache_db/cache.db
CACHE_TTL_SECONDS=86400        # 24 hours
MAX_PARALLEL_TOOLS=5
MAX_RETRIES=3
TIMEOUT_SECONDS=300

# ============================================
# API Server (Optional)
# ============================================
PORT=8080                      # Cloud Run uses this

# ============================================
# Rate Limiting
# ============================================
# Groq: 100K tokens/day, 12K tokens/minute
# Gemini: 10 requests/minute (free tier)
```

### Provider Comparison

| Feature | Groq | Gemini |
|---------|------|--------|
| **Model** | llama-3.3-70b-versatile | gemini-2.0-flash-exp |
| **Speed** | Very Fast (LPU) | Fast |
| **Cost** | Free tier: 100K tokens/day | Free tier: 10 RPM |
| **Rate Limit** | 12K tokens/minute | 10 requests/minute |
| **Function Calling** | Native | Native (custom schema) |
| **Best For** | Development, high volume | Production, reliability |

### Switching Providers

```python
# Method 1: Environment variable
export LLM_PROVIDER=gemini

# Method 2: Code
agent = DataScienceCopilot(provider="gemini")
```

## 📊 Example Outputs

### Dataset Profile
```json
{
  "shape": {"rows": 891, "columns": 12},
  "memory_usage": {"total_mb": 0.08},
  "column_types": {
    "numeric": ["Age", "Fare", "SibSp", "Parch"],
    "categorical": ["Sex", "Embarked", "Cabin"],
    "datetime": []
  },
  "missing_values_per_column": {
    "Age": {"count": 177, "percentage": 19.87},
    "Cabin": {"count": 687, "percentage": 77.1}
  },
  "unique_counts_per_column": {
    "PassengerId": 891,
    "Survived": 2,
    "Pclass": 3,
    "Sex": 2
  },
  "overall_stats": {
    "null_percentage": 19.87,
    "duplicate_rows": 0
  }
}
```

### Data Quality Issues
```json
{
  "critical": [
    {
      "type": "missing_values",
      "message": "Column 'Age' has 177 missing values (19.87%)",
      "severity": "high"
    }
  ],
  "warning": [
    {
      "type": "outliers",
      "message": "Column 'Fare' has 8 outliers using IQR method",
      "severity": "medium"
    }
  ],
  "info": [
    {
      "type": "high_cardinality",
      "message": "Column 'Cabin' has 147 unique values (16.5% of dataset)",
      "severity": "low"
    }
  ]
}
```

### Model Training Results
```json
{
  "status": "success",
  "best_model": {
    "name": "xgboost",
    "score": 0.8156,
    "model_path": "./outputs/models/xgboost_20251223_143022.pkl"
  },
  "models": {
    "logistic_regression": {
      "test_metrics": {
        "accuracy": 0.7821,
        "f1": 0.7234,
        "precision": 0.7456,
        "recall": 0.7023
      },
      "train_time_seconds": 0.45
    },
    "random_forest": {
      "test_metrics": {
        "accuracy": 0.8045,
        "f1": 0.7589,
        "precision": 0.7712,
        "recall": 0.7468
      },
      "train_time_seconds": 1.23
    },
    "xgboost": {
      "test_metrics": {
        "accuracy": 0.8156,
        "f1": 0.7692,
        "precision": 0.7879,
        "recall": 0.7527
      },
      "train_time_seconds": 0.89,
      "feature_importance": {
        "Sex_male": 0.234,
        "Fare": 0.189,
        "Age": 0.156
      }
    }
  },
  "execution_time_seconds": 45.67,
  "api_calls_made": 12
}
```

### Workflow History
```json
{
  "workflow_history": [
    {
      "step": 1,
      "tool": "profile_dataset",
      "status": "success",
      "execution_time": 2.34
    },
    {
      "step": 2,
      "tool": "detect_data_quality_issues",
      "status": "success",
      "execution_time": 1.89
    },
    {
      "step": 3,
      "tool": "clean_missing_values",
      "status": "success",
      "output": "./outputs/data/cleaned.csv",
      "execution_time": 3.12
    },
    {
      "step": 4,
      "tool": "encode_categorical",
      "status": "success",
      "output": "./outputs/data/encoded.csv",
      "execution_time": 1.56
    },
    {
      "step": 5,
      "tool": "train_baseline_models",
      "status": "success",
      "best_model": "xgboost",
      "score": 0.8156,
      "execution_time": 15.23
    }
  ]
}
```

## 🚀 Deployment

### Local Development
```bash
# CLI
python src/cli.py analyze data.csv --target price

# Web UI
python chat_ui.py

# API Server
cd src/api && python app.py
```

### Google Cloud Run Deployment

#### Prerequisites
- Google Cloud account
- `gcloud` CLI installed
- Docker installed

#### Step 1: Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY .env .env

# Expose port
EXPOSE 8080

# Run API server
CMD ["python", "src/api/app.py"]
```

#### Step 2: Deploy to Cloud Run
```bash
# Build and deploy
gcloud run deploy data-science-agent \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GROQ_API_KEY=your_key,LLM_PROVIDER=groq

# Get service URL
gcloud run services describe data-science-agent --region us-central1
```

#### Step 3: Test deployment
```bash
# Get service URL from previous command
SERVICE_URL=$(gcloud run services describe data-science-agent \
  --region us-central1 --format 'value(status.url)')

# Test health endpoint
curl $SERVICE_URL/health

# Run analysis
curl -X POST $SERVICE_URL/run \
  -F "file=@data.csv" \
  -F "task_description=Predict house prices" \
  -F "target_col=price"
```

### Environment Variables for Cloud Run
```bash
# Set during deployment
gcloud run deploy data-science-agent \
  --set-env-vars \
    GROQ_API_KEY=your_groq_key \
    GOOGLE_API_KEY=your_google_key \
    LLM_PROVIDER=groq \
    CACHE_DB_PATH=/tmp/cache.db
```

### Production Considerations
- **Stateless**: Cloud Run instances are ephemeral
- **Temporary Storage**: Use `/tmp` for files (10GB limit)
- **Cold Starts**: Agent initialization (~2-3 seconds)
- **Scaling**: Auto-scales based on traffic
- **Costs**: Pay per request (free tier: 2M requests/month)

## 🧪 Development

### Running Tests
```bash
# Install dev dependencies
pip install pytest pytest-cov pytest-mock

# Run all tests
pytest

# With coverage
pytest --cov=src tests/

# Specific test
pytest tests/test_tools.py -v
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Tools

1. **Create tool function** in appropriate module (e.g., `src/tools/custom_tools.py`)
```python
def my_custom_tool(file_path: str, param: str) -> Dict[str, Any]:
    """
    Tool description for LLM.
    
    Args:
        file_path: Path to dataset
        param: Custom parameter
        
    Returns:
        Dictionary with results
    """
    # Implementation
    return {"status": "success", "result": "..."}
```

2. **Register in tools_registry.py**
```python
TOOLS.append({
    "type": "function",
    "function": {
        "name": "my_custom_tool",
        "description": "Brief description for LLM",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "param": {"type": "string"}
            },
            "required": ["file_path", "param"]
        }
    }
})
```

3. **Add to orchestrator.py**
```python
from tools.custom_tools import my_custom_tool

# In _build_tool_functions_map():
"my_custom_tool": my_custom_tool,
```

## 🎓 Key Concepts

### 1. Function Calling
The LLM analyzes user requests and automatically selects appropriate tools with correct parameters.

```python
# User: "Clean missing values in age column using median"
# LLM generates:
{
  "tool": "clean_missing_values",
  "arguments": {
    "file_path": "data.csv",
    "strategy": {"Age": "median"},
    "output_path": "./outputs/data/cleaned.csv"
  }
}
```

### 2. Tool Chaining
Output of one tool becomes input for the next, creating intelligent workflows.

```
profile_dataset(data.csv)
    ↓
detect_data_quality_issues(data.csv)
    ↓
clean_missing_values(data.csv) → cleaned.csv
    ↓
encode_categorical(cleaned.csv) → encoded.csv
    ↓
train_baseline_models(encoded.csv) → models
```

### 3. Caching
Expensive operations are cached with 24-hour TTL to avoid recomputation.

```python
# First call: Computes and caches
profile_dataset("data.csv")  # Takes 2 seconds

# Second call: Returns from cache
profile_dataset("data.csv")  # Takes 0.01 seconds
```

### 4. Error Recovery
Automatic retry with parameter correction when tools fail.

```python
# ❌ First attempt fails
train_baseline_models(target_col="magnitude")
# Error: "Column 'magnitude' not found. Did you mean 'mag'?"

# ✅ Agent automatically retries
train_baseline_models(target_col="mag")  # Success!
```

### 5. Token Optimization
Smart compression to stay within API limits.

**Before**: 12,440 tokens (exceeds 12K limit)  
**After**: 6,045 tokens (50% of limit) with all 75 tools available

### 6. Intent Detection
LLM distinguishes between analysis vs. modeling requests.

```python
# Analysis only - No model training
"Analyze this dataset and show insights"  
→ Profile + EDA + Visualizations

# Full ML pipeline - Includes training
"Predict house prices using this data"
→ Profile + Clean + Engineer + Train + Report
```

## 📚 Advanced Features

### Auto ML Pipeline
Zero-configuration end-to-end workflow:

```python
from src.tools.auto_pipeline import auto_ml_pipeline

result = auto_ml_pipeline(
    file_path="data.csv",
    target_col="price",
    task_type="auto",  # Detects classification/regression
    feature_engineering_level="advanced"
)

# Automatically:
# 1. Detects column types
# 2. Cleans missing values
# 3. Handles outliers
# 4. Engineers features
# 5. Selects best features
# 6. Returns processed data
```

### Interactive EDA Reports

```python
from src.tools.eda_reports import generate_combined_eda_report

# Generates both Sweetviz + ydata-profiling reports
result = generate_combined_eda_report(
    file_path="data.csv",
    target_column="price",
    output_dir="./outputs/reports"
)

# Creates:
# - sweetviz_report.html (beautiful, fast)
# - ydata_profiling_report.html (comprehensive, detailed)
```

### Business Intelligence

```python
from src.tools.business_intelligence import perform_rfm_analysis

# RFM customer segmentation
result = perform_rfm_analysis(
    file_path="customers.csv",
    customer_id_col="customer_id",
    date_col="purchase_date",
    monetary_col="amount"
)

# Returns:
# - RFM scores per customer
# - Segment labels (Champions, At Risk, etc.)
# - Actionable recommendations
```

### Model Explainability

```python
from src.tools.production_mlops import explain_predictions

# SHAP explanations for predictions
result = explain_predictions(
    model_path="./outputs/models/xgboost.pkl",
    data_path="data.csv",
    instance_index=0,  # Explain first row
    method="shap"
)

# Returns:
# - Feature contributions
# - SHAP values
# - Force plot data
# - Waterfall plot data
```

## 🔮 Roadmap

### ✅ Completed (Current Version)
- [x] 75 specialized data science tools
- [x] Dual LLM support (Groq + Gemini)
- [x] Token optimization (34% reduction)
- [x] CLI, Web UI, and REST API interfaces
- [x] SQLite caching system
- [x] Interactive Plotly visualizations
- [x] Automated EDA reports (Sweetviz, ydata-profiling)
- [x] Error recovery and retry logic
- [x] Cloud Run deployment support
- [x] Advanced feature engineering
- [x] MLOps tools (drift detection, explainability)

### 🚧 In Progress
- [ ] Google Cloud integration (BigQuery, Vertex AI, Cloud Storage)
- [ ] Execution backends (local vs. cloud adapters)
- [ ] Multi-dataset workflows
- [ ] SQL database connectivity

### 📋 Planned Features

#### Phase 1: Cloud Integration
- [ ] BigQuery backend for large datasets
- [ ] Vertex AI for model training at scale
- [ ] Cloud Storage for artifact management
- [ ] Cloud Logging integration
- [ ] Configuration-driven cloud/local switching

#### Phase 2: Advanced Workflows
- [ ] Multi-step pipeline orchestration
- [ ] Parallel dataset processing
- [ ] SQL database support (PostgreSQL, MySQL)
- [ ] Real-time streaming data support
- [ ] Scheduled workflow execution

#### Phase 3: Kaggle Integration
- [ ] Direct Kaggle API integration
- [ ] Automated submission pipeline
- [ ] Competition-specific strategies
- [ ] Leaderboard tracking
- [ ] Ensemble voting across submissions

#### Phase 4: Collaboration Features
- [ ] Shared workspace support
- [ ] Version control for workflows
- [ ] Team collaboration tools
- [ ] Workflow templates marketplace
- [ ] Model registry and versioning

#### Phase 5: Enhanced Intelligence
- [ ] Multi-agent collaboration
- [ ] Conversation memory across sessions
- [ ] Learning from past workflows
- [ ] Automated hyperparameter tuning
- [ ] Neural architecture search

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Improvement
1. **More Tools**: Time series forecasting, recommendation systems, reinforcement learning
2. **Better Prompts**: Improve LLM reasoning quality and tool selection
3. **Performance**: Optimize tool execution speed and memory usage
4. **Documentation**: More examples, tutorials, and use cases
5. **Testing**: Increase test coverage and add integration tests
6. **Cloud Integration**: Help build cloud backend adapters

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Standards
- **Style**: Follow PEP 8, use Black for formatting
- **Type Hints**: Add type annotations to all functions
- **Documentation**: Docstrings for all public functions
- **Testing**: Maintain >80% test coverage
- **Tools**: Each tool should be independent and composable

## � Troubleshooting

### Common Issues

#### 1. API Key Errors
```bash
# Error: "API key must be provided"
# Solution: Set environment variable
export GROQ_API_KEY=your_key_here
# OR
export GOOGLE_API_KEY=your_key_here
```

#### 2. Rate Limit Errors
```bash
# Error: "Rate limit exceeded"
# Solution: Switch provider or adjust rate limiting
export LLM_PROVIDER=gemini  # Has different rate limits
```

#### 3. Token Limit Errors
```bash
# Error: "Request too large for model"
# Solution: Token optimization is automatic, but you can:
# - Use smaller datasets for initial testing
# - Reduce max_iterations parameter
# - Clear cache: python src/cli.py clear-cache
```

#### 4. Memory Issues
```python
# Error: "MemoryError" or system freezing
# Solution: Use Polars instead of Pandas (already default)
# For very large files:
result = agent.analyze(
    file_path="huge_data.csv",
    task_description="...",
    use_cache=True  # Enables caching to reduce recomputation
)
```

#### 5. Import Errors
```bash
# Error: "ModuleNotFoundError: No module named 'src'"
# Solution: Run from project root
cd Data-Science-Agent
python src/cli.py analyze data.csv

# OR add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 6. Cache Issues
```bash
# Error: Cache database corrupted
# Solution: Clear and reinitialize cache
python src/cli.py clear-cache
# OR manually delete
rm -rf cache_db/
```

### Getting Help
- **GitHub Issues**: [Report bugs](https://github.com/Surfing-Ninja/Data-Science-Agent/issues)
- **Discussions**: [Ask questions](https://github.com/Surfing-Ninja/Data-Science-Agent/discussions)
- **Documentation**: Check this README and code comments

## 📊 Performance Benchmarks

### Dataset Sizes Tested
- **Small**: < 1MB, < 10K rows → ~5-10 seconds
- **Medium**: 1-10MB, 10K-100K rows → ~20-60 seconds
- **Large**: 10-100MB, 100K-1M rows → ~2-5 minutes
- **Very Large**: > 100MB → Consider BigQuery backend (coming soon)

### Tool Execution Times (Average)
| Tool | Small Dataset | Large Dataset |
|------|---------------|---------------|
| profile_dataset | 0.5s | 3s |
| clean_missing_values | 1s | 8s |
| encode_categorical | 0.8s | 5s |
| train_baseline_models | 5s | 30s |
| generate_eda_plots | 3s | 12s |
| hyperparameter_tuning | 30s | 5min |

### Memory Usage
- **Agent Initialization**: ~200MB
- **Small Dataset**: +50MB
- **Medium Dataset**: +200MB
- **Large Dataset**: +1GB
- **Peak (with training)**: +2GB

## 📝 License

MIT License

Copyright (c) 2025 Data Science Agent Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🙏 Acknowledgments

- **Groq**: Blazing-fast LLM inference with LPU architecture
- **Google**: Gemini API and cloud infrastructure
- **Polars Team**: Incredible high-performance dataframe library
- **Kaggle**: Datasets, competitions, and inspiration
- **Open Source Community**: NumPy, scikit-learn, XGBoost, and countless other libraries
- **Sweetviz & ydata-profiling**: Automated EDA report generation
- **Gradio**: Simple and beautiful web UI framework
- **FastAPI**: Modern, fast web framework for APIs

## 📧 Contact & Support

- **Repository**: [github.com/Surfing-Ninja/Data-Science-Agent](https://github.com/Surfing-Ninja/Data-Science-Agent)
- **Issues**: [Report bugs or request features](https://github.com/Surfing-Ninja/Data-Science-Agent/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/Surfing-Ninja/Data-Science-Agent/discussions)
- **Owner**: [@Surfing-Ninja](https://github.com/Surfing-Ninja)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

**Built with ❤️ for the data science community**

*"Making data science accessible through AI automation"*

---

### Quick Links
- [Installation](#-quick-start)
- [Features](#-features)
- [Documentation](#-key-concepts)
- [Examples](./examples/)
- [API Reference](#-usage-options)
- [Contributing](#-contributing)
- [License](#-license)
