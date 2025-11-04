# Data Science Copilot 🤖

An AI-powered CLI tool that automates end-to-end data science workflows using Groq's GPT-OSS-120B. Think "Cursor for Data Science" - an intelligent agent that profiles data, cleans datasets, engineers features, and trains models autonomously.

## 🎯 Project Goal

Achieve **50-70th percentile performance** on Kaggle competitions through intelligent automation of data science workflows, proving that AI agents can handle real-world ML tasks end-to-end.

## ✨ Features

### Intelligent Orchestration
- **Native Groq Function Calling**: Direct integration with Groq's GPT-OSS-120B (no frameworks)
- **Smart Routing**: LLM intelligently selects and chains tools based on task requirements
- **Adaptive Workflows**: Automatically adjusts strategy based on data characteristics
- **Context-Aware**: Remembers previous steps and learns from tool outputs

### Comprehensive Tool Suite

#### 📊 Data Profiling
- `profile_dataset`: Complete dataset statistics, types, memory usage
- `detect_data_quality_issues`: Outlier detection, duplicates, inconsistencies
- `analyze_correlations`: Feature relationships and target correlations

#### 🧹 Data Cleaning
- `clean_missing_values`: Smart imputation (median/mean/mode/forward_fill)
- `handle_outliers`: IQR-based detection with clip/winsorize/remove
- `fix_data_types`: Auto-detect and fix incorrect types

#### 🔧 Feature Engineering
- `create_time_features`: Extract temporal patterns (cyclical encoding)
- `encode_categorical`: One-hot, target, and frequency encoding

#### 🤖 Model Training
- `train_baseline_models`: Train and compare LR, RF, XGBoost
- `generate_model_report`: Metrics, feature importance, SHAP values

### Performance Optimization
- **SQLite Caching**: Memoization of expensive operations
- **Polars & DuckDB**: Fast data processing
- **Parallel Execution**: Independent tools run concurrently
- **Streaming Responses**: Real-time LLM output

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Groq API key ([Get one here](https://console.groq.com))

### Installation

```bash
# Clone the repository
cd datascience-copilot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Basic Usage

```bash
# Complete analysis workflow
python src/cli.py analyze data.csv --target Survived --task "Predict survival"

# Quick profile
python src/cli.py profile data.csv

# Clean dataset
python src/cli.py clean data.csv --output cleaned_data.csv

# Train models
python src/cli.py train cleaned_data.csv Survived --task-type classification

# Cache management
python src/cli.py cache-stats
python src/cli.py clear-cache
```

### Example Workflow

```python
from src.orchestrator import DataScienceCopilot

# Initialize
copilot = DataScienceCopilot(reasoning_effort="medium")

# Run complete workflow
result = copilot.analyze(
    file_path="titanic.csv",
    task_description="Predict passenger survival with feature engineering",
    target_col="Survived"
)

print(result["summary"])
print(f"Best Model: {result['workflow_history']}")
```

See `examples/titanic_example.py` for a complete example.

## 🏗️ Architecture

### Design Philosophy
- **No Frameworks**: Pure Groq SDK function calling (NO LangChain/CrewAI/LangGraph)
- **Single Orchestrator**: One intelligent router instead of multi-agent complexity
- **Fast Iteration**: Optimize for debugging and rapid development
- **Actionable Results**: Every tool returns structured, LLM-parseable outputs

### System Flow

```
User Query
    ↓
Orchestrator (GPT-OSS-120B)
    ↓
Function Calling Decision
    ↓
Tool Execution (Parallel where possible)
    ↓
Result Synthesis
    ↓
Final Recommendations
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Groq GPT-OSS-120B | Function calling & reasoning |
| Data Processing | Polars | Fast dataframe operations |
| SQL Operations | DuckDB | Complex queries |
| ML Libraries | scikit-learn, XGBoost | Model training |
| Hyperparameter Tuning | Optuna | Optimization |
| Explainability | SHAP | Feature importance |
| Caching | SQLite | Memoization |
| CLI | Typer + Rich | User interface |

## 📁 Project Structure

```
datascience-copilot/
├── src/
│   ├── orchestrator.py          # Main DataScienceCopilot class
│   ├── cli.py                   # CLI interface
│   ├── tools/
│   │   ├── data_profiling.py    # Dataset analysis tools
│   │   ├── data_cleaning.py     # Cleaning & preprocessing
│   │   ├── feature_engineering.py # Feature creation
│   │   ├── model_training.py    # ML training & evaluation
│   │   └── tools_registry.py    # Groq function definitions
│   ├── cache/
│   │   └── cache_manager.py     # SQLite caching
│   └── utils/
│       ├── polars_helpers.py    # Data manipulation utilities
│       └── validation.py        # Input validation
├── examples/
│   └── titanic_example.py       # Complete workflow demo
├── tests/
│   ├── test_tools.py
│   └── test_orchestrator.py
├── data/                         # Test datasets
├── outputs/                      # Generated outputs
│   ├── models/                   # Trained models
│   ├── reports/                  # Analysis reports
│   └── data/                     # Cleaned datasets
├── cache_db/                     # SQLite cache
├── requirements.txt
├── .env.example
└── README.md
```

## 🔧 Configuration

Edit `.env` file:

```env
# Groq API
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
REASONING_EFFORT=medium  # low, medium, high

# Cache
CACHE_DB_PATH=./cache_db/cache.db
CACHE_TTL_SECONDS=86400  # 24 hours

# Performance
MAX_PARALLEL_TOOLS=5
MAX_RETRIES=3
TIMEOUT_SECONDS=300
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
  "overall_stats": {
    "null_percentage": 19.87,
    "duplicate_rows": 0
  }
}
```

### Model Training Results
```json
{
  "best_model": {
    "name": "xgboost",
    "score": 0.8156,
    "model_path": "./outputs/models/xgboost.pkl"
  },
  "models": {
    "xgboost": {
      "test_metrics": {
        "accuracy": 0.8156,
        "f1": 0.7692,
        "precision": 0.7879,
        "recall": 0.7527
      }
    }
  }
}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_tools.py -v
```

## 🎓 Learning Resources

### Understanding the Architecture

**Why No Frameworks?**
- Direct control over function calling behavior
- Easier debugging and iteration
- Faster execution (no abstraction overhead)
- Better understanding of LLM capabilities

**Why Groq?**
- Extremely fast inference (LPU architecture)
- Native function calling support
- Cost-effective for high-volume usage
- `reasoning_effort` parameter for quality control

**Why Polars over Pandas?**
- 10-100x faster for large datasets
- Better memory efficiency
- Native parallelization
- Lazy evaluation

### Key Concepts

1. **Function Calling**: LLM decides which tools to use and with what parameters
2. **Tool Chaining**: Output of one tool becomes input for next
3. **Caching**: Avoid re-computing expensive operations
4. **Streaming**: Show progress in real-time

## 🔮 Roadmap

### Phase 2: Advanced Features
- [ ] Optuna hyperparameter tuning integration
- [ ] AutoML model selection
- [ ] Cross-validation strategies
- [ ] Ensemble methods

### Phase 3: Kaggle Integration
- [ ] Direct Kaggle API integration
- [ ] Automated submission pipeline
- [ ] Competition-specific strategies
- [ ] Leaderboard tracking

### Phase 4: Production Features
- [ ] REST API server
- [ ] Web UI dashboard
- [ ] Multi-dataset workflows
- [ ] Collaborative features

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **More Tools**: Time series, NLP preprocessing, image features
2. **Better Prompts**: Improve LLM reasoning quality
3. **Performance**: Optimize tool execution speed
4. **Tests**: Increase coverage
5. **Documentation**: More examples and tutorials

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Groq for blazing-fast LLM inference
- Polars team for incredible data processing library
- Kaggle community for datasets and competitions
- OpenAI for function calling paradigm

## 📧 Support

- Issues: [GitHub Issues](https://github.com/yourusername/datascience-copilot/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/datascience-copilot/discussions)

---

**Built with ❤️ for the data science community**

*"Making data science accessible through AI automation"*
