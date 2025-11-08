# EDA Report Generation - Quick Start Guide

## 🚀 Installation

Install the required libraries:

```powershell
# Install EDA report generation libraries
pip install sweetviz==2.3.1 ydata-profiling==4.6.4 plotly==5.18.0

# Or install all requirements
pip install -r requirements.txt
```

## ✅ Verify Installation

Test that the libraries are properly installed:

```powershell
python -c "import sweetviz; import ydata_profiling; import plotly; print('✅ All EDA libraries installed successfully!')"
```

## 🧪 Quick Test

### Test 1: Sweetviz Report

Create a test script `test_sweetviz.py`:

```python
from src.tools.eda_reports import generate_sweetviz_report
import pandas as pd
import os

# Create sample data
data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'income': [50000, 60000, 75000, 80000, 90000, 95000, 100000, 110000],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA', 'Chicago', 'NYC', 'LA'],
    'satisfaction': [4, 5, 3, 4, 5, 4, 3, 5]
}
df = pd.DataFrame(data)
os.makedirs('./temp', exist_ok=True)
df.to_csv('./temp/test_data.csv', index=False)

# Generate report
result = generate_sweetviz_report(
    file_path='./temp/test_data.csv',
    output_path='./outputs/reports/test_sweetviz.html',
    target_column='satisfaction'
)

print(result)
```

Run it:
```powershell
python test_sweetviz.py
```

Expected output:
```json
{
    "success": true,
    "report_path": "./outputs/reports/test_sweetviz.html",
    "message": "✅ Sweetviz report generated successfully",
    "summary": {
        "features": 4,
        "rows": 8,
        "numeric_features": 2,
        "categorical_features": 2,
        "missing_percentage": 0.0,
        "target_column": "satisfaction",
        "has_comparison": false
    }
}
```

Then open `./outputs/reports/test_sweetviz.html` in your browser!

---

### Test 2: ydata-profiling Report

Create a test script `test_profiling.py`:

```python
from src.tools.eda_reports import generate_ydata_profiling_report
import pandas as pd
import os

# Create sample data
data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'income': [50000, 60000, 75000, 80000, 90000, 95000, 100000, 110000],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA', 'Chicago', 'NYC', 'LA'],
    'satisfaction': [4, 5, 3, 4, 5, 4, 3, 5]
}
df = pd.DataFrame(data)
os.makedirs('./temp', exist_ok=True)
df.to_csv('./temp/test_data.csv', index=False)

# Generate report (minimal mode for faster generation)
result = generate_ydata_profiling_report(
    file_path='./temp/test_data.csv',
    output_path='./outputs/reports/test_profiling.html',
    minimal=True,
    title='Test Data Profile'
)

print(result)
```

Run it:
```powershell
python test_profiling.py
```

Expected output:
```json
{
    "success": true,
    "report_path": "./outputs/reports/test_profiling.html",
    "message": "✅ ydata-profiling report generated successfully",
    "statistics": {
        "dataset_size": {
            "rows": 8,
            "columns": 4,
            "cells": 32
        },
        "variable_types": {
            "numeric": 2,
            "categorical": 2,
            "boolean": 0
        },
        "data_quality": {
            "missing_cells": 0,
            "missing_percentage": 0.0,
            "duplicate_rows": 0
        }
    }
}
```

Then open `./outputs/reports/test_profiling.html` in your browser!

---

### Test 3: Combined Reports

Create a test script `test_combined.py`:

```python
from src.tools.eda_reports import generate_combined_eda_report
import pandas as pd
import os

# Create sample data
data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'income': [50000, 60000, 75000, 80000, 90000, 95000, 100000, 110000],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA', 'Chicago', 'NYC', 'LA'],
    'satisfaction': [4, 5, 3, 4, 5, 4, 3, 5]
}
df = pd.DataFrame(data)
os.makedirs('./temp', exist_ok=True)
df.to_csv('./temp/test_data.csv', index=False)

# Generate both reports
result = generate_combined_eda_report(
    file_path='./temp/test_data.csv',
    output_dir='./outputs/reports/combined_test',
    target_column='satisfaction',
    minimal=True
)

print(result)
```

Run it:
```powershell
python test_combined.py
```

Expected: Both reports generated in `./outputs/reports/combined_test/`

---

## 🎯 Test with Real Data

Once installation is verified, test with the Gradio UI:

1. Start the UI:
```powershell
python chat_ui.py
```

2. Upload a dataset

3. Ask the agent:
   - "Generate a Sweetviz report"
   - "Create a comprehensive profiling report"
   - "Give me both EDA reports"

4. Check `./outputs/reports/` for generated HTML files

---

## 🐛 Troubleshooting

### Issue: "Import 'sweetviz' could not be resolved"
**Solution**: Run `pip install sweetviz==2.3.1`

### Issue: "Import 'ydata_profiling' could not be resolved"
**Solution**: Run `pip install ydata-profiling==4.6.4`

### Issue: Reports take too long to generate
**Solution**: Use `minimal=True` for ydata-profiling:
```python
generate_ydata_profiling_report(file_path, minimal=True)
```

### Issue: "Module 'pandas' has no attribute 'Int64Dtype'"
**Solution**: Update pandas: `pip install --upgrade pandas`

### Issue: Missing plotly dependency
**Solution**: `pip install plotly==5.18.0`

---

## 📊 Expected Output Locations

After running the tests, you should see:

```
./outputs/reports/
├── test_sweetviz.html          ← Sweetviz test output
├── test_profiling.html         ← Profiling test output
└── combined_test/
    ├── sweetviz_report.html    ← Combined: Sweetviz
    └── ydata_profile.html      ← Combined: Profiling
```

Open any HTML file in your browser to view the interactive reports!

---

## ✅ Success Criteria

You've successfully installed and tested the EDA report tools when:

- ✅ All pip install commands complete without errors
- ✅ Import verification shows "All EDA libraries installed successfully!"
- ✅ Test scripts run without errors
- ✅ HTML reports are generated in `./outputs/reports/`
- ✅ Reports open properly in your browser
- ✅ Agent can call the tools via Gradio UI

---

## 🎉 Ready to Use!

Once all tests pass, the EDA report generation feature is fully operational. The agent can now generate professional-grade reports with a single command!

**Next Step**: Try SQL Database Support implementation
