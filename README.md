# Data Science Agent 🤖

A production-grade **autonomous AI agent** for end-to-end data science workflows. Upload datasets, describe your goal in natural language, and let the AI handle profiling, cleaning, feature engineering, model training, and visualization.

**Key Differentiator**: Not just a chatbot - a true AI agent with 75+ specialized tools, intelligent orchestration, dual LLM support, session memory, code interpreter, and Cloud Run API.

## 🎯 Project Vision

Build an **autonomous data science system** that achieves **50-70th percentile performance** on Kaggle competitions through intelligent automation, proving AI agents can handle real-world ML workflows end-to-end.

---

## ✨ Core Features

### **🤖 Intelligent Agent System**
- **82+ Specialized Tools** across 11 categories (profiling, cleaning, feature engineering, ML, visualization, BigQuery)
- **Dual LLM Support**: Groq (llama-3.3-70b) + Google Gemini (2.0-flash-exp)
- **Smart Orchestration**: LLM-powered function calling with intelligent tool chaining
- **Session Memory**: Contextual awareness across conversations ("cross-validate it", "try with Ridge")
- **Code Interpreter**: Write and execute custom Python code for tasks beyond predefined tools
- **Error Recovery**: Automatic retry with corrected parameters
- **Reasoning Modules**: Dedicated LLM reasoning layer with 19 specialized functions
- **Cloud Integration**: BigQuery data access + GCS artifact storage

### 🎨 **Multiple Interfaces**
- **Gradio Web UI** (`chat_ui.py`): Upload files, chat interface, visual plots
- **CLI Interface** (`src/cli.py`): Command-line workflow automation
- **REST API** (`src/api/app.py`): Cloud Run-ready FastAPI wrapper
- **Python SDK**: Direct programmatic access

### 📊 **Complete ML Pipeline**
1. **Data Profiling** → Statistics, types, quality issues
2. **Data Cleaning** → Smart imputation, outlier handling, type conversion
3. **Feature Engineering** → Time features, encoding, interactions, ratios
4. **Model Training** → XGBoost, LightGBM, CatBoost, ensemble methods
5. **Hyperparameter Tuning** → Optuna-based optimization
6. **Visualization** → Matplotlib, Plotly, interactive dashboards
7. **EDA Reports** → Sweetviz, ydata-profiling HTML reports
8. **Explainability** → SHAP values, feature importance

### ⚡ **Performance & Scale**
- **Token Optimization**: 34% reduction in LLM context (compressed tool schemas)
- **SQLite Caching**: Memoization of expensive operations with TTL
- **Polars & DuckDB**: 10-100x faster than pandas for large datasets
- **Rate Limiting**: Intelligent API call management (Groq: 12K TPM, Gemini: 10 RPM)
- **Cloud Ready**: FastAPI service for Google Cloud Run deployment

---

## 🏗️ Architecture

### **System Design**

```
┌─────────────────────────────────────────────────────────────┐
│                   User Interfaces                            │
│  Gradio UI  │  CLI  │  REST API  │  Python SDK               │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              DataScienceCopilot Orchestrator                 │
│  • LLM Function Calling (Groq/Gemini)                       │
│  • Session Memory Management                                 │
│  • Tool Execution & Chaining                                 │
│  • Error Recovery & Retry Logic                              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    75+ Specialized Tools                     │
│  Data Profiling │ Cleaning │ Feature Engineering             │
│  Model Training │ Visualization │ EDA Reports                │
│  NLP/Text │ Computer Vision │ Time Series │ MLOps           │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Execution & Storage Backends                    │
│  Local: Polars, sklearn, XGBoost                            │
│  Cloud: BigQuery, Vertex AI, Cloud Storage (planned)        │
│  Cache: SQLite with TTL                                      │
└─────────────────────────────────────────────────────────────┘
```

### **Tech Stack**

| Layer | Technologies |
|-------|-------------|
| **LLM** | Groq (llama-3.3-70b), Google Gemini (2.0-flash-exp) |
| **Data Processing** | Polars, DuckDB, Pandas, PyArrow, BigQuery |
| **ML/AI** | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **EDA Reports** | Sweetviz, ydata-profiling |
| **Explainability** | SHAP, LIME |
| **APIs** | FastAPI, Uvicorn |
| **UI** | Gradio, Typer + Rich (CLI) |
| **Storage** | SQLite (cache), CSV, Parquet, Google Cloud Storage |
| **Cloud** | Google Cloud Run, BigQuery, GCS, Vertex AI (planned) |

---

## 🚀 Quick Start

### **Prerequisites**
- Python 3.9+
- API Keys: [Groq](https://console.groq.com) or [Google AI Studio](https://makersuite.google.com/app/apikey)

### **Installation**

```bash
# Clone repository
git clone https://github.com/Surfing-Ninja/Data-Science-Agent.git
cd Data-Science-Agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
# GROQ_API_KEY=your_groq_key
# GOOGLE_API_KEY=your_google_key (optional)
# LLM_PROVIDER=groq  # or "gemini"
```

### **Usage Examples**

#### **1. Gradio Web UI** (Recommended for beginners)
```bash
python chat_ui.py
# Opens at http://localhost:7860
# Upload CSV → Ask: "Analyze this data and predict house prices"
```

#### **2. CLI Interface**
```bash
# Complete workflow
python src/cli.py analyze data.csv --target price --task "Predict house prices"

# Quick profiling
python src/cli.py profile data.csv

# Train models only
python src/cli.py train cleaned.csv Survived --task-type classification
```

#### **3. Python SDK**
```python
from src.orchestrator import DataScienceCopilot

# Initialize agent
agent = DataScienceCopilot(
    provider="groq",  # or "gemini"
    reasoning_effort="medium"
)

# Run workflow
result = agent.analyze(
    file_path="titanic.csv",
    task_description="Build a model to predict passenger survival",
    target_col="Survived"
)

print(f"Status: {result['status']}")
print(f"Best Model: {result['best_model']}")
print(f"Accuracy: {result['best_score']}")
```

#### **4. REST API** (Cloud Run Ready)
```bash
# Start local server
cd src/api
python app.py
# Server runs at http://localhost:8080

# Make API call
curl -X POST http://localhost:8080/run \
  -F "file=@data.csv" \
  -F "task_description=Analyze and predict churn" \
  -F "target_col=churn"
```

---

## 📁 Project Structure

```
Data-Science-Agent/
├── src/
│   ├── orchestrator.py              # Main agent brain (1,136 lines)
│   ├── cli.py                       # CLI interface (346 lines)
│   ├── api/
│   │   └── app.py                   # FastAPI Cloud Run wrapper (331 lines)
│   ├── bigquery/                    # BigQuery integration 🆕
│   │   ├── __init__.py             # BigQuery tools (4 functions)
│   │   └── client.py               # BigQuery client wrapper
│   ├── storage/                     # Artifact storage 🆕
│   │   ├── artifact_store.py       # Local + GCS backends (613 lines)
│   │   └── helpers.py              # Storage helper functions (125 lines)
│   ├── reasoning/                   # LLM reasoning layer 🆕
│   │   ├── __init__.py             # Core reasoning engine (350 lines)
│   │   ├── data_understanding.py   # Data insights (6 functions)
│   │   ├── model_explanation.py    # Model interpretation (6 functions)
│   │   └── business_summary.py     # Business translations (7 functions)
│   ├── cache/
│   │   └── cache_manager.py        # SQLite caching with TTL
│   ├── tools/                       # 82+ specialized tools
│   │   ├── data_profiling.py       # Dataset analysis
│   │   ├── data_cleaning.py        # Cleaning & preprocessing
│   │   ├── feature_engineering.py  # Feature creation
│   │   ├── model_training.py       # ML training
│   │   ├── visualization_engine.py # Matplotlib/Seaborn plots
│   │   ├── plotly_visualizations.py # Interactive charts
│   │   ├── eda_reports.py          # Sweetviz, ydata-profiling
│   │   ├── advanced_*.py           # Advanced features
│   │   └── tools_registry.py       # All 82 tool definitions (1,600+ lines)
│   └── utils/                       # Helper utilities
│       ├── polars_helpers.py       # Data manipulation
│       └── validation.py           # Input validation
├── chat_ui.py                       # Gradio web interface (912 lines)
├── examples/
│   └── titanic_example.py           # Complete workflow demo
├── outputs/
│   ├── data/                        # Processed datasets
│   ├── models/                      # Trained models (.pkl)
│   ├── plots/                       # Visualizations (.png, .html)
│   └── reports/                     # EDA reports (.html)
├── cache_db/                        # SQLite cache storage
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
└── README.md                        # This file
```

---

## 🛠️ Tool Categories (82 Tools Total)

### **📊 Data Profiling & Analysis (7 tools)**
- `profile_dataset`, `detect_data_quality_issues`, `analyze_correlations`, `get_smart_summary`, `compare_datasets`, `calculate_statistics`, `detect_skewness`

### **☁️ BigQuery Integration (4 tools)** 🆕
- `bigquery_profile_table`, `bigquery_load_table`, `bigquery_execute_query`, `bigquery_write_results`

### **🧹 Data Cleaning (8 tools)**
- `clean_missing_values`, `handle_outliers`, `remove_duplicates`, `filter_rows`, `rename_columns`, `drop_columns`, `sort_data`, `fix_data_types`

### **🔧 Feature Engineering (13 tools)**
- `encode_categorical`, `force_numeric_conversion`, `smart_type_inference`, `create_time_features`, `create_interaction_features`, `create_aggregation_features`, `create_ratio_features`, `create_statistical_features`, `create_log_features`, `create_binned_features`, `engineer_text_features`, `auto_feature_engineering`, `auto_feature_selection`

### **🤖 Model Training & Tuning (6 tools)**
- `train_baseline_models`, `hyperparameter_tuning`, `train_ensemble_models`, `perform_cross_validation`, `generate_model_report`, `auto_ml_pipeline`

### **📈 Visualization (11 tools)**
- `generate_all_plots`, `generate_data_quality_plots`, `generate_eda_plots`, `generate_model_performance_plots`, `generate_feature_importance_plot`, `generate_interactive_scatter`, `generate_interactive_histogram`, `generate_interactive_correlation_heatmap`, `generate_interactive_box_plots`, `generate_interactive_time_series`, `generate_plotly_dashboard`

### **📊 EDA Reports (3 tools)**
- `generate_sweetviz_report`, `generate_ydata_profiling_report`, `generate_combined_eda_report`

### **🔬 Advanced Analysis (11 tools)**
- `perform_eda_analysis`, `detect_model_issues`, `detect_anomalies`, `detect_and_handle_multicollinearity`, `perform_statistical_tests`, `analyze_root_cause`, `detect_trends_and_seasonality`, `detect_anomalies_advanced`, `perform_hypothesis_testing`, `analyze_distribution`, `perform_segment_analysis`

### **📝 Data Wrangling (3 tools)**
- `merge_datasets`, `concat_datasets`, `reshape_dataset`

### **🚀 MLOps & Production (5 tools)**
- `monitor_model_drift`, `explain_predictions`, `generate_model_card`, `perform_ab_test_analysis`, `detect_feature_leakage`

### **⏰ Time Series (3 tools)**
- `forecast_time_series`, `detect_seasonality_trends`, `create_time_series_features`

### **💼 Business Intelligence (4 tools)**
- `perform_cohort_analysis`, `perform_rfm_analysis`, `detect_causal_relationships`, `generate_business_insights`

### **📚 NLP/Text (4 tools)**
- `perform_topic_modeling`, `perform_named_entity_recognition`, `analyze_sentiment_advanced`, `perform_text_similarity`

### **🖼️ Computer Vision (3 tools)**
- `extract_image_features`, `perform_image_clustering`, `analyze_tabular_image_hybrid`

---

## 🎯 Advanced Features

### **1. Session Memory**
The agent remembers context across conversations:

```python
# Conversation 1
"Train a model on earthquake.csv to predict magnitude"
→ Agent trains XGBoost, achieves 0.92 R²

# Conversation 2 (Same session)
"Cross-validate it"
→ Agent knows: model=XGBoost, dataset=earthquake.csv, target=magnitude
→ Runs 5-fold CV automatically
```

### **2. Code Interpreter**
Execute custom Python code for tasks beyond predefined tools:

```python
User: "Make a Plotly scatter with custom dropdown filters"

Agent: execute_python_code(code='''
import plotly.graph_objects as go
df = pd.read_csv('./temp/data.csv')
# Custom visualization code...
fig.write_html('./outputs/code/custom_plot.html')
''')
```

### **3. Token Optimization**
System stays under LLM token limits even with 75 tools:

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| Tool Schemas | 8,193 tokens | 5,463 tokens | 34% |
| Tool Results | 5,000+ tokens | 50-200 tokens | 90%+ |

### **4. Error Recovery**
Agent learns from errors and auto-corrects:

```python
# Attempt 1
train_baseline_models(target_col="magnitude")
→ Error: Column 'magnitude' not found. Hint: Did you mean 'mag'?

# Attempt 2 (Automatic)
train_baseline_models(target_col="mag")
→ Success! Trained 4 models, best: XGBoost (0.92 R²)
```

---

## ☁️ Cloud Features

### **1. BigQuery Integration** 🆕
Direct access to BigQuery tables without local downloads:

```python
# Profile a BigQuery table
agent.chat("Profile the table project.dataset.sales")

# Query and analyze
agent.chat("Query top 10 customers by revenue from BigQuery")

# Write results back
agent.chat("Write the cleaned data to BigQuery table project.dataset.sales_clean")
```

**Available Tools:**
- `bigquery_profile_table`: Get statistics for any BigQuery table
- `bigquery_load_table`: Load BigQuery data into local Polars DataFrame
- `bigquery_execute_query`: Run SQL queries directly on BigQuery
- `bigquery_write_results`: Write processed data back to BigQuery

**Setup:**
```bash
# Install BigQuery dependencies
pip install google-cloud-bigquery db-dtypes

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

**Looker-Compatible Schemas:**

The project defines stable BigQuery table schemas for BI tools (see [`BIGQUERY_SCHEMAS.md`](BIGQUERY_SCHEMAS.md)):
- 📊 `model_metrics` - Model performance tracking over time
- 🎯 `feature_importance` - Feature impact analysis
- 🔮 `predictions` - Prediction monitoring with actuals
- 📋 `data_profile_summary` - Data quality metrics

**Design Principles:**
- Stable schemas (no breaking changes without versioning)
- Consistent snake_case naming
- Clear dimension/metric separation
- Dashboard-ready with sample Looker views

### **2. Artifact Storage** 🆕
Unified storage abstraction - switch between local and GCS with zero code changes:

```python
# Local storage (default)
agent.save_model(model, "my_model.pkl")  
# → Saves to outputs/models/my_model.pkl

# GCS storage (automatic when GCS credentials present)
agent.save_model(model, "my_model.pkl")
# → Saves to gs://your-bucket/models/my_model_v1.pkl with versioning
```

**Features:**
- **Automatic Backend Selection**: Uses GCS if credentials available, falls back to local
- **Versioning**: Automatic version suffixes for GCS artifacts
- **Metadata**: Stores creation time, size, checksums
- **Unified API**: Same code works for local and cloud storage

**Setup:**
```bash
# Install GCS dependencies
pip install google-cloud-storage

# Set bucket (optional, defaults to local)
export GCS_BUCKET="your-gcs-bucket-name"
```

### **3. Reasoning Modules** 🆕
Dedicated LLM reasoning layer with clear boundaries (no raw data access, no training decisions):

```python
from reasoning.data_understanding import explain_dataset
from reasoning.model_explanation import explain_model_performance
from reasoning.business_summary import create_executive_summary

# Data insights
insights = explain_dataset(summary={
    "rows": 10000,
    "columns": 20,
    "missing_values": {"age": {"count": 150, "percentage": 1.5}}
})

# Model explanations
explanation = explain_model_performance(metrics={
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.88
}, task_type="classification")

# Business summaries
summary = create_executive_summary(
    project_results={"model_accuracy": 0.95},
    project_name="churn_prediction",
    business_objective="Reduce customer churn"
)
```

**19 Reasoning Functions:**
- **Data Understanding**: explain_dataset, suggest_transformations, identify_feature_engineering_opportunities, explain_missing_values, compare_datasets (6 functions)
- **Model Explanation**: explain_model_performance, interpret_feature_importance, diagnose_model_failure, explain_prediction, compare_models, explain_overfitting (6 functions)
- **Business Summary**: create_executive_summary, estimate_business_impact, create_stakeholder_report, translate_technical_to_business, prioritize_next_steps, explain_to_customer, assess_deployment_readiness (7 functions)

**Design Principles:**
- ✅ **NO Raw Data Access**: Only summaries/statistics allowed
- ✅ **NO Training Decisions**: Only explanations, never execution
- ✅ **Structured Output**: JSON schemas for cacheability
- ✅ **Dual Backend**: Works with both Gemini and Groq

---

## 🔧 Configuration

### **Environment Variables** (`.env`)

#### Step 3: Test deployment
```bash
# LLM Provider
LLM_PROVIDER=groq               # "groq" or "gemini"
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key  # Optional

# Model Selection
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_MODEL=gemini-2.0-flash-exp
REASONING_EFFORT=medium         # low, medium, high

# Cache Settings
CACHE_DB_PATH=./cache_db/cache.db
CACHE_TTL_SECONDS=86400         # 24 hours

# Cloud Features (Optional)
GCS_BUCKET=your-gcs-bucket-name                           # For artifact storage
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-key.json  # For BigQuery + GCS

# Cloud Run (for API deployment)
PORT=8080
```

### **Provider Comparison**

| Feature | Groq | Gemini |
|---------|------|--------|
| **Model** | llama-3.3-70b-versatile | gemini-2.0-flash-exp |
| **Speed** | ⚡ Extremely fast (LPU) | 🚀 Very fast |
| **Free Tier** | 100K tokens/day | 1,500 requests/day |
| **Rate Limit** | 12K tokens/min | 10 requests/min |
| **Best For** | High-volume, low-latency | Free tier, high quota |

---

## 🚀 Cloud Deployment (Google Cloud Run)

### **Deploy REST API**

```bash
# 1. Build Docker image (Dockerfile provided)
docker build -t data-science-agent .

# 2. Push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/data-science-agent

# 3. Deploy to Cloud Run
gcloud run deploy data-science-agent \
  --image gcr.io/PROJECT_ID/data-science-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --timeout 3600 \
  --set-env-vars GROQ_API_KEY=your_key,LLM_PROVIDER=groq

# 4. Test deployment
curl -X POST https://your-service-url/run \
  -F "file=@data.csv" \
  -F "task_description=Predict churn"
```

### **API Endpoints**

- `GET /` - Health check
- `GET /health` - Readiness probe
- `POST /run` - Full analysis workflow
- `POST /profile` - Quick dataset profiling
- `GET /tools` - List all available tools

---

## 🗺️ Roadmap

### **Phase 1: Core Agent** ✅ COMPLETE
- [x] 75 specialized tools
- [x] Dual LLM support (Groq + Gemini)
- [x] CLI + Gradio UI
- [x] SQLite caching
- [x] Token optimization

### **Phase 2: Intelligence** ✅ COMPLETE
- [x] Session memory
- [x] Code interpreter
- [x] Error recovery
- [x] EDA reports (Sweetviz, ydata-profiling)
- [x] Interactive Plotly visualizations

### **Phase 3: Cloud Native** ✅ COMPLETE
- [x] FastAPI Cloud Run wrapper with 4 REST endpoints
- [x] BigQuery integration (4 tools: profile, load, query, write)
- [x] Artifact Storage abstraction (Local ↔ GCS switching)
- [x] Reasoning modules for LLM explanations (19 functions)
- [x] Looker-compatible BigQuery schemas (4 stable tables)
- [ ] Vertex AI model training (planned)
- [ ] Cloud Logging & Monitoring (planned)

### **Phase 4: Enterprise** 📋 PLANNED
- [ ] Multi-user authentication
- [ ] Team workspaces
- [ ] Model registry
- [ ] Automated retraining pipelines

### **Phase 5: Kaggle Integration** 🎯 FUTURE
- [ ] Direct Kaggle API integration
- [ ] Automated competition workflow
- [ ] Ensemble strategies
- [ ] Submission automation

---

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

1. **New Tools**: Time series forecasting, NLP preprocessing, image augmentation
2. **Cloud Backends**: AWS, Azure support
3. **Performance**: Optimize tool execution, reduce latency
4. **UI/UX**: Better visualization, workflow builder
5. **Documentation**: Tutorials, video guides, blog posts

---

## 📜 License

MIT License - See LICENSE file for details

---

## 📧 Support & Community

- **Issues**: [GitHub Issues](https://github.com/Surfing-Ninja/Data-Science-Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Surfing-Ninja/Data-Science-Agent/discussions)

---

## 📊 Project Stats

- **Lines of Code**: ~18,000+
- **Tools**: 82 specialized functions (75 core + 4 BigQuery + 3 storage helpers)
- **Reasoning Functions**: 19 LLM-powered explanation modules
- **Supported Models**: 10+ (LR, Ridge, Lasso, RF, XGBoost, LightGBM, CatBoost, etc.)
- **Visualization Types**: 20+ (static + interactive)
- **Data Formats**: CSV, Parquet, JSON, BigQuery tables
- **Cloud Platforms**: Google Cloud (Run, BigQuery, GCS) - AWS/Azure planned

---

<div align="center">

**Built with ❤️ for the Data Science Community**

*"Making data science accessible through AI automation"*

⭐ Star this repo if you find it useful! ⭐

</div>
