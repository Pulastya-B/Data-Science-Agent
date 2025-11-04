# 🚀 Quick Installation & Testing Guide

## ✅ **All 44 Tools Are Complete!**

---

## 📦 **Installation**

### **Step 1: Core Dependencies (Required)**
```bash
pip install -r requirements.txt
```

This installs all essential dependencies including:
- Groq API, Polars, scikit-learn, XGBoost
- Optuna, SHAP, LIME, Prophet
- Plotly, Rich, statsmodels
- Basic NLP (TextBlob) and CV (Pillow, OpenCV)

### **Step 2: Optional NLP Tools (Recommended)**
```bash
# For advanced topic modeling, NER, sentiment analysis
pip install spacy transformers sentence-transformers bertopic

# Download spaCy language model
python -m spacy download en_core_web_sm
```

**Enables:**
- `perform_topic_modeling` with BERTopic
- `perform_named_entity_recognition` with spaCy
- `analyze_sentiment_advanced` with transformers
- `perform_text_similarity` with semantic embeddings

### **Step 3: Optional Computer Vision (Recommended for CV)**
```bash
# For CNN-based image features
pip install torch torchvision
```

**Enables:**
- `extract_image_features` with ResNet50/EfficientNet/VGG16
- Deep learning embeddings for `perform_image_clustering`
- Advanced features for `analyze_tabular_image_hybrid`

### **Step 4: Optional Business Intelligence**
```bash
# For specialized analytics
pip install lifetimes econml
```

**Enables:**
- Customer lifetime value modeling in `perform_cohort_analysis`
- Advanced causal inference in `detect_causal_relationships`

---

## 🧪 **Quick Tests**

### **Test 1: Basic Tool Import**
```python
# Test if tools can be imported
from tools.data_profiling import profile_dataset
from tools.advanced_training import hyperparameter_tuning
from tools.production_mlops import monitor_model_drift
from tools.time_series import forecast_time_series
from tools.nlp_text_analytics import perform_topic_modeling
from tools.business_intelligence import perform_rfm_analysis
from tools.computer_vision import extract_image_features

print("✅ All tools imported successfully!")
```

### **Test 2: Simple Data Profiling**
```python
import polars as pl
from tools.data_profiling import profile_dataset

# Create sample data
data = pl.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 75000, 90000, 100000],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA']
})

# Profile the data
result = profile_dataset(data)
print(result)
```

### **Test 3: Topic Modeling (NLP)**
```python
import polars as pl
from tools.nlp_text_analytics import perform_topic_modeling

# Sample text data
texts = [
    "Machine learning is fascinating",
    "Deep learning models are powerful",
    "Natural language processing is useful",
    "Computer vision detects objects",
    "Data science involves statistics"
]

data = pl.DataFrame({'text': texts * 10})  # Repeat for more data

# Perform topic modeling
result = perform_topic_modeling(
    data=data,
    text_column='text',
    n_topics=2,
    method='lda'
)

print(f"Found {len(result['topics'])} topics")
for topic in result['topics']:
    print(f"Topic {topic['topic_id']}: {', '.join(topic['words'][:5])}")
```

### **Test 4: RFM Analysis (Business Intelligence)**
```python
import polars as pl
from datetime import datetime, timedelta
from tools.business_intelligence import perform_rfm_analysis

# Sample transaction data
dates = [datetime.now() - timedelta(days=i*10) for i in range(20)]
data = pl.DataFrame({
    'customer_id': ['C1', 'C2', 'C3', 'C1', 'C2'] * 4,
    'date': dates,
    'amount': [100, 200, 150, 300, 250] * 4
})

# Perform RFM analysis
result = perform_rfm_analysis(
    data=data,
    customer_id_column='customer_id',
    date_column='date',
    value_column='amount'
)

print(f"Total customers: {result['total_customers']}")
print(f"Segments: {list(result['segment_summary'].keys())}")
```

### **Test 5: Time Series Forecasting**
```python
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from tools.time_series import forecast_time_series

# Generate sample time series
dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
values = np.sin(np.arange(100) * 0.1) * 50 + 100 + np.random.randn(100) * 5

data = pl.DataFrame({
    'date': dates,
    'value': values
})

# Forecast
result = forecast_time_series(
    data=data,
    date_column='date',
    value_column='value',
    method='arima',
    forecast_periods=10
)

print(f"Forecast for next {result['forecast_periods']} periods:")
print(result['forecast'][:5])
```

---

## 🎯 **Tool Categories & File Locations**

### **1. Data Profiling & Quality** (`data_profiling.py`)
- `profile_dataset()` - Dataset overview
- `detect_data_quality_issues()` - Quality checks
- `analyze_correlations()` - Correlation analysis

### **2. Data Cleaning** (`data_cleaning.py`)
- `clean_missing_values()` - Missing value imputation
- `handle_outliers()` - Outlier detection & removal
- `fix_data_types()` - Type conversion

### **3. Feature Engineering** (`feature_engineering.py`, `advanced_feature_engineering.py`)
- `create_time_features()` - Temporal features
- `encode_categorical()` - Categorical encoding
- `create_interaction_features()` - Feature interactions
- `create_aggregation_features()` - Aggregations
- `engineer_text_features()` - Text features
- `auto_feature_engineering()` - LLM-powered features

### **4. Model Training** (`model_training.py`, `advanced_training.py`)
- `train_baseline_models()` - Quick model comparison
- `generate_model_report()` - Model evaluation
- `hyperparameter_tuning()` - Optuna optimization
- `train_ensemble_models()` - Stacking/blending
- `perform_cross_validation()` - CV with OOF

### **5. Preprocessing** (`advanced_preprocessing.py`)
- `handle_imbalanced_data()` - SMOTE, ADASYN
- `perform_feature_scaling()` - Scaling methods
- `split_data_strategically()` - Smart data splits

### **6. Analysis & Diagnostics** (`advanced_analysis.py`)
- `perform_eda_analysis()` - Interactive EDA
- `detect_model_issues()` - Overfitting detection
- `detect_anomalies()` - Anomaly detection
- `detect_and_handle_multicollinearity()` - VIF analysis
- `perform_statistical_tests()` - Hypothesis testing

### **7. Production & MLOps** (`production_mlops.py`)
- `monitor_model_drift()` - Drift detection
- `explain_predictions()` - SHAP, LIME
- `generate_model_card()` - Model documentation
- `perform_ab_test_analysis()` - A/B testing
- `detect_feature_leakage()` - Leakage detection

### **8. Time Series** (`time_series.py`)
- `forecast_time_series()` - ARIMA, Prophet
- `detect_seasonality_trends()` - STL decomposition
- `create_time_series_features()` - Temporal features

### **9. NLP & Text** (`nlp_text_analytics.py`)
- `perform_topic_modeling()` - LDA, NMF, BERTopic
- `perform_named_entity_recognition()` - NER
- `analyze_sentiment_advanced()` - Sentiment & emotions
- `perform_text_similarity()` - Text similarity

### **10. Business Intelligence** (`business_intelligence.py`)
- `perform_cohort_analysis()` - Retention, churn
- `perform_rfm_analysis()` - Customer segmentation
- `detect_causal_relationships()` - Causal inference
- `generate_business_insights()` - LLM insights

### **11. Computer Vision** (`computer_vision.py`)
- `extract_image_features()` - CNN embeddings
- `perform_image_clustering()` - Image clustering
- `analyze_tabular_image_hybrid()` - Multi-modal

---

## 📋 **Common Issues & Solutions**

### **Issue 1: Import Errors**
```
ImportError: No module named 'optuna'
```
**Solution:** Install missing dependency
```bash
pip install optuna
```

### **Issue 2: spaCy Model Not Found**
```
OSError: [E050] Can't find model 'en_core_web_sm'
```
**Solution:** Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

### **Issue 3: Prophet Installation Issues on Windows**
```
ERROR: Failed building wheel for prophet
```
**Solution:** Install with conda
```bash
conda install -c conda-forge prophet
```

### **Issue 4: PyTorch Installation**
```
# CPU-only (smaller, faster install)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU support (requires CUDA)
pip install torch torchvision
```

### **Issue 5: Groq API Key Not Found**
```
ValueError: Groq API key not found
```
**Solution:** Set environment variable
```bash
# Windows PowerShell
$env:GROQ_API_KEY="your-api-key-here"

# Or create .env file
echo GROQ_API_KEY=your-api-key-here > .env
```

---

## 🔄 **Workflow Examples**

### **Kaggle Competition Workflow**
```python
from tools.data_profiling import profile_dataset
from tools.advanced_preprocessing import handle_imbalanced_data
from tools.advanced_feature_engineering import create_interaction_features
from tools.advanced_training import hyperparameter_tuning, train_ensemble_models

# 1. Profile data
profile = profile_dataset(train_data)

# 2. Handle imbalanced data
balanced_data = handle_imbalanced_data(
    train_data, target_column='target', method='smote'
)

# 3. Create features
features = create_interaction_features(
    balanced_data['data'], method='polynomial', degree=2
)

# 4. Tune hyperparameters
best_params = hyperparameter_tuning(
    features, target_column='target', model_type='xgboost'
)

# 5. Train ensemble
ensemble = train_ensemble_models(
    features, target_column='target', method='stacking'
)
```

### **Production ML Workflow**
```python
from tools.production_mlops import monitor_model_drift, explain_predictions

# Monitor for drift
drift = monitor_model_drift(
    reference_data=train_data,
    current_data=production_data,
    feature_columns=features,
    target_column='target'
)

if drift['drift_detected']:
    print("⚠️ Model drift detected!")
    
# Explain predictions
explanations = explain_predictions(
    model=trained_model,
    data=test_sample,
    method='shap'
)
```

### **Business Analytics Workflow**
```python
from tools.business_intelligence import (
    perform_cohort_analysis,
    perform_rfm_analysis,
    generate_business_insights
)

# Cohort analysis
cohorts = perform_cohort_analysis(
    transactions_df,
    customer_id_column='customer_id',
    date_column='date',
    value_column='amount',
    metric='retention'
)

# RFM segmentation
rfm = perform_rfm_analysis(
    transactions_df,
    customer_id_column='customer_id',
    date_column='date',
    value_column='amount'
)

# Generate insights
insights = generate_business_insights(
    data=transactions_df,
    analysis_type='rfm',
    analysis_results=rfm
)

print(insights['insights_summary'])
```

---

## 📚 **Documentation Files**

1. `ALL_TOOLS_COMPLETE.md` - Complete tool listing & features
2. `ADVANCED_TOOLS_SUMMARY.md` - Advanced ML tools guide
3. `INDUSTRY_TOOLS_STATUS.md` - Industry tools implementation details
4. `FINAL_IMPLEMENTATION_SUMMARY.md` - Overall project summary
5. `QUICK_START.md` - This file

---

## 🎯 **Next Steps**

1. **Test Core Tools** ✅
   - Run the quick tests above
   - Verify imports work
   - Test with your own data

2. **Install Optional Dependencies** (as needed)
   - NLP: `pip install spacy transformers bertopic`
   - CV: `pip install torch torchvision`
   - BI: `pip install lifetimes econml`

3. **Integration** (Future Work)
   - Update `tools/__init__.py`
   - Update `tools_registry.py`
   - Update `orchestrator.py`
   - Update `cli.py`

4. **Create Examples**
   - Kaggle competition notebook
   - Production ML pipeline
   - Business analytics dashboard
   - NLP analysis example
   - CV workflow example

---

## 💪 **You're Ready to Go!**

You now have **44 production-ready data science tools** covering:
- ✅ Kaggle competitions
- ✅ Production ML systems
- ✅ Time series forecasting
- ✅ NLP & text analytics
- ✅ Business intelligence
- ✅ Computer vision
- ✅ Advanced analytics

**Start testing and building amazing data science solutions!** 🚀
