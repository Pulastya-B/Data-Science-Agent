# Code Interpreter Quick Reference

## 🎯 When to Use

Use `execute_python_code` when the user asks for:

### ✅ USE Code Interpreter
- **Custom visualizations**: "Plot with dropdown filter", "Multi-panel dashboard", "Animated chart"
- **Specific Plotly features**: Custom buttons, sliders, range selectors, updatemenus
- **Domain-specific calculations**: Custom formulas, business metrics
- **Unique transformations**: Complex pivots, custom aggregations
- **Custom export formats**: Specific JSON structure, specialized HTML

### ❌ DON'T Use Code Interpreter
- **Standard EDA plots**: Use `generate_eda_plots()`
- **Interactive dashboards** (generic): Use `generate_plotly_dashboard()`
- **Training models**: Use `train_baseline_models()`
- **Data cleaning**: Use `clean_missing_values()`, `handle_outliers()`
- **Feature engineering**: Use encoding/scaling tools

---

## 📝 Code Template

```python
execute_python_code(code='''
# Load data
import pandas as pd
df = pd.read_csv('./temp/your_data.csv')

# Your custom code here
# ...

# Save output
output.write_html('./outputs/code/result.html')
# OR
df.to_csv('./outputs/code/result.csv', index=False)

# Print success message
print("✅ Saved to: ./outputs/code/result.html")
''', working_directory="./outputs/code")
```

---

## 🎨 Common Use Cases

### 1. Plotly Dropdown Filter
```python
code = '''
import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv('./temp/sales.csv')

fig = go.Figure()

# Add traces for each category
for category in df['category'].unique():
    cat_data = df[df['category'] == category]
    fig.add_trace(go.Scatter(
        x=cat_data['date'],
        y=cat_data['sales'],
        name=category,
        mode='lines+markers'
    ))

# Add dropdown
fig.update_layout(
    updatemenus=[{
        'buttons': [
            {'label': 'All', 'method': 'update',
             'args': [{'visible': [True] * len(df['category'].unique())}]},
        ] + [
            {'label': cat, 'method': 'update',
             'args': [{'visible': [cat == c for c in df['category'].unique()]}]}
            for cat in df['category'].unique()
        ],
        'direction': 'down',
        'showactive': True
    }]
)

fig.write_html('./outputs/code/sales_with_dropdown.html')
print("✅ Chart saved")
'''
execute_python_code(code=code)
```

### 2. Multi-Panel Dashboard
```python
code = '''
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

df = pd.read_csv('./temp/data.csv')

# Create 2x2 grid
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Sales Trend', 'Category Distribution', 
                   'Top Products', 'Regional Performance')
)

# Add different plot types to each panel
fig.add_trace(go.Scatter(...), row=1, col=1)
fig.add_trace(go.Bar(...), row=1, col=2)
fig.add_trace(go.Pie(...), row=2, col=1)
fig.add_trace(go.Heatmap(...), row=2, col=2)

fig.update_layout(height=800, showlegend=True)
fig.write_html('./outputs/code/dashboard.html')
'''
execute_python_code(code=code)
```

### 3. Animated Timeline
```python
code = '''
import plotly.express as px
import pandas as pd

df = pd.read_csv('./temp/timeseries.csv')

fig = px.scatter(
    df, 
    x='metric_x', 
    y='metric_y',
    animation_frame='year',
    animation_group='product',
    size='sales',
    color='category',
    hover_name='product',
    range_x=[0, 100],
    range_y=[0, 100]
)

fig.write_html('./outputs/code/animated.html')
'''
execute_python_code(code=code)
```

### 4. Custom Calculation
```python
code = '''
import pandas as pd
import numpy as np

df = pd.read_csv('./temp/customers.csv')

# Custom business logic
df['customer_score'] = (
    df['recency'].rank(pct=True) * 0.3 +
    df['frequency'].rank(pct=True) * 0.4 +
    df['monetary'].rank(pct=True) * 0.3
) * 100

# Add tier classification
df['tier'] = pd.cut(df['customer_score'], 
                    bins=[0, 33, 66, 100],
                    labels=['Bronze', 'Silver', 'Gold'])

df.to_csv('./outputs/code/scored_customers.csv', index=False)
print(f"✅ Scored {len(df)} customers")
'''
execute_python_code(code=code)
```

---

## 🔒 Security

### ✅ Allowed
- pandas, numpy, matplotlib, seaborn, plotly operations
- File I/O in working directory
- JSON, CSV, HTML export
- Mathematical operations

### ❌ Blocked
- `subprocess` calls
- `eval()` / `exec()` / `compile()`
- `__import__` magic
- Network requests (no requests/urllib)

---

## 📊 Auto-Imported Libraries

Every code execution automatically has:

```python
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
```

No need to import these!

---

## 🎯 File Paths

### Reading Data
- User uploads → `./temp/filename.csv`
- Processed data → `./outputs/data/cleaned.csv`, `./outputs/data/encoded.csv`

### Writing Outputs
- **Plots**: `./outputs/code/your_plot.html`
- **Data**: `./outputs/code/your_data.csv`
- **Reports**: `./outputs/code/your_report.html`

---

## 💡 Tips

1. **Always save outputs to files** - Don't just display in memory
2. **Print success messages** - User needs to know where files are saved
3. **Use descriptive filenames** - `bike_sales_dropdown.html` not `plot1.html`
4. **Handle errors gracefully** - Use try/except if needed
5. **Keep code concise** - Aim for <50 lines when possible

---

## 🚀 Example Requests

| User Request | Code Interpreter Needed? |
|--------------|-------------------------|
| "Plot sales by month with dropdown to filter by product" | ✅ YES - Custom dropdown |
| "Create animated chart showing growth over years" | ✅ YES - Animation |
| "Build dashboard with 4 panels showing different metrics" | ✅ YES - Custom layout |
| "Calculate CLV = (avg_purchase × frequency × lifespan)" | ✅ YES - Custom formula |
| "Generate scatter plot" | ❌ NO - Use generate_interactive_scatter |
| "Create histogram of age distribution" | ❌ NO - Use generate_interactive_histogram |
| "Train a regression model" | ❌ NO - Use train_baseline_models |

---

## 📖 Full Documentation

See `CODE_INTERPRETER_FEATURE.md` for comprehensive guide including:
- Detailed use cases
- Security safeguards
- Return format specifications
- Before/After comparisons
- Testing examples
