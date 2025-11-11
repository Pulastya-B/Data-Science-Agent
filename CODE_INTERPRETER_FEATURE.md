# 🚀 Code Interpreter - True AI Agent Capability

## The Problem You Identified

**Current Limitation:**
```
User: "Make a plot of extended sales by month for each bike model with a Plotly dropdown filter"
Agent: ❌ "I don't have a tool for that specific visualization"
```

**Why This Happens:**
The agent can only use **predefined tools**. Even with 75+ tools, it can't handle:
- Custom visualizations (specific chart types, custom filters, unique layouts)
- Domain-specific calculations not covered by existing tools
- Unique data transformations
- Interactive widgets with custom logic

**This is the difference between:**
- ❌ **Function Calling Bot**: "I can only do what my 75 tools allow"
- ✅ **True AI Agent**: "I can write code to solve ANY problem"

---

## The Solution: Code Interpreter

**New Tool: `execute_python_code()`**

This tool lets the AI agent **write and execute custom Python code** for ANY task not covered by existing tools.

### How It Works

```python
# User Request:
"Make a plot of extended sales by month for each bike model with a Plotly dropdown filter"

# Agent Response (Internally):
execute_python_code(code='''
import plotly.graph_objects as go
import pandas as pd

# Load the data
df = pd.read_csv('./temp/sales_data.csv')

# Create line plot with all bike models
fig = go.Figure()

for bike_model in df['bike_model'].unique():
    model_data = df[df['bike_model'] == bike_model]
    fig.add_trace(go.Scatter(
        x=model_data['month'],
        y=model_data['extended_sales'],
        name=bike_model,
        mode='lines+markers'
    ))

# Add dropdown filter
fig.update_layout(
    title='Extended Sales by Month for Each Bike Model',
    xaxis_title='Month-Year',
    yaxis_title='Total Extended Sales',
    updatemenus=[{
        'buttons': [
            {'label': 'All Models', 'method': 'update',
             'args': [{'visible': [True] * len(df['bike_model'].unique())}]},
        ] + [
            {'label': model, 'method': 'update',
             'args': [{'visible': [model == m for m in df['bike_model'].unique()]}]}
            for model in df['bike_model'].unique()
        ],
        'direction': 'down',
        'showactive': True,
        'x': 0.1,
        'y': 1.15
    }]
)

# Save to file
fig.write_html('./outputs/code/bike_sales_interactive.html')
print("✅ Interactive chart saved to: ./outputs/code/bike_sales_interactive.html")
''', working_directory="./outputs/code")

# User sees:
"✅ Created interactive plot with dropdown filter at: ./outputs/code/bike_sales_interactive.html"
```

---

## Features

### 1. **Auto-Imported Libraries**
Every code execution automatically imports:
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

### 2. **Security Safeguards**
- ❌ Blocks dangerous operations: `subprocess`, `eval()`, `exec()`, `compile()`
- ✅ Allows safe operations: file I/O, data processing, plotting
- ⏱️ 60-second timeout (configurable)
- 📁 Sandboxed working directory

### 3. **Smart Execution**
- Captures stdout and stderr
- Tracks generated files
- Returns detailed execution summary
- Cleans up temp files automatically

---

## Use Cases

### ✅ When to Use Code Interpreter

| User Request | Use Code Interpreter? | Why |
|--------------|----------------------|-----|
| "Plot with dropdown filter by bike model" | ✅ YES | Specific interactive widget not in generic tools |
| "Create multi-panel dashboard with custom layout" | ✅ YES | Custom layout requirements |
| "Animated timeline of sales over years" | ✅ YES | Animation not in standard tools |
| "Calculate customer lifetime value with custom formula" | ✅ YES | Domain-specific calculation |
| "Export data in custom JSON format" | ✅ YES | Specific export format |
| "Train a baseline model" | ❌ NO | Use `train_baseline_models()` tool |
| "Clean missing values" | ❌ NO | Use `clean_missing_values()` tool |
| "Generate EDA plots" | ❌ NO | Use `generate_eda_plots()` tool |

### Examples of Custom Tasks

**1. Interactive Plotly Dashboard with Custom Controls**
```python
execute_python_code(code='''
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

df = pd.read_csv('./temp/data.csv')

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Sales Trend', 'Distribution', 'Top Models', 'Geography')
)

# Add custom traces with specific styling
# Add range slider, custom buttons, annotations
# ...

fig.write_html('./outputs/code/custom_dashboard.html')
''')
```

**2. Domain-Specific Calculation**
```python
execute_python_code(code='''
import pandas as pd

df = pd.read_csv('./temp/customers.csv')

# Calculate RFM score with custom weights
df['rfm_score'] = (
    df['recency'].rank(pct=True) * 0.3 +
    df['frequency'].rank(pct=True) * 0.4 +
    df['monetary'].rank(pct=True) * 0.3
)

df.to_csv('./outputs/code/rfm_scored.csv', index=False)
print(f"✅ Scored {len(df)} customers")
''')
```

**3. Animated Visualization**
```python
execute_python_code(code='''
import plotly.express as px
import pandas as pd

df = pd.read_csv('./temp/sales_over_time.csv')

fig = px.scatter(
    df, x='sales', y='profit', animation_frame='year',
    animation_group='product', size='quantity', color='category',
    hover_name='product', range_x=[0, 100000], range_y=[0, 50000]
)

fig.write_html('./outputs/code/animated_sales.html')
''')
```

---

## System Prompt Integration

The agent now knows WHEN to use code interpreter vs existing tools:

```
⭐ CODE INTERPRETER - WHEN TO USE:
Use execute_python_code for custom tasks NOT covered by existing tools:
1. Custom Visualizations: Specific plot types, interactive widgets
2. Domain-Specific Calculations: Custom metrics, formulas
3. Custom Data Transformations: Unique reshaping, pivoting
4. Interactive Widgets: Plotly dropdowns, sliders, buttons
5. Export Formats: Custom HTML, JSON, specialized files

When User Says:
- "Make a plot with dropdown to filter..." → execute_python_code
- "Create an interactive dashboard with..." → execute_python_code
- "Show sales by month with filter by model" → execute_python_code
- "Calculate [domain-specific metric]" → execute_python_code
```

---

## Tool Definition

```json
{
  "type": "function",
  "function": {
    "name": "execute_python_code",
    "description": "⭐ CRITICAL TOOL - Execute custom Python code for ANY data science task not covered by existing tools. This is what makes you a TRUE AI AGENT. Use when user requests custom visualizations, domain-specific calculations, or unique transformations.",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {
          "type": "string",
          "description": "Python code to execute. Auto-imported: pandas, polars, numpy, matplotlib, seaborn, plotly. Save outputs to files."
        },
        "working_directory": {
          "type": "string",
          "description": "Directory to run code (default: ./outputs/code)"
        },
        "timeout": {
          "type": "integer",
          "description": "Max execution time in seconds (default: 60)"
        }
      },
      "required": ["code"]
    }
  }
}
```

---

## Return Format

```python
{
    "success": True,
    "stdout": "✅ Interactive chart saved to: ./outputs/code/bike_sales.html\n",
    "stderr": None,
    "message": "✅ Code executed successfully",
    "generated_files": [
        "outputs/code/bike_sales_interactive.html"
    ],
    "working_directory": "./outputs/code",
    "execution_summary": {
        "lines_of_code": 42,
        "files_generated": 1
    }
}
```

---

## Comparison: Before vs After

### Before (Function Calling Bot)
```
User: "Create a plot with dropdown filter by bike model"
Agent: "I can create a basic plot with generate_plotly_dashboard(), 
        but it won't have the specific dropdown you want. 
        You'll need to modify the code manually."
Result: ❌ User has to code it themselves
```

### After (True AI Agent)
```
User: "Create a plot with dropdown filter by bike model"
Agent: *writes custom Plotly code with dropdown*
       execute_python_code(code='''
       # Custom Plotly code with exact dropdown implementation
       ''')
Result: ✅ Exact visualization user requested, fully automated
```

---

## Files Modified

1. **`src/tools/code_interpreter.py`** (NEW - 300 lines)
   - `execute_python_code()` - Main code execution engine
   - `execute_code_from_file()` - Run existing scripts
   - Security validation
   - Auto-import of common libraries

2. **`src/tools/tools_registry.py`**
   - Added 2 new tool definitions
   - Total tools: 75 → **77**

3. **`src/tools/__init__.py`**
   - Imported code interpreter functions
   - Updated `__all__` exports

4. **`src/orchestrator.py`**
   - Imported code interpreter tools
   - Registered in function map
   - **Enhanced system prompt** with code interpreter usage guidelines

---

## Testing

### Test 1: Custom Plotly Plot
```python
from src.copilot import run_copilot

run_copilot(
    user_query="Make a plot of extended sales by month for each bike model with a Plotly dropdown filter",
    csv_path="./temp/sales_data.csv"
)

# Expected: Creates interactive HTML with dropdown at ./outputs/code/
```

### Test 2: Domain-Specific Calculation
```python
run_copilot(
    user_query="Calculate customer lifetime value as: (avg_purchase * purchase_frequency * customer_lifespan) - acquisition_cost",
    csv_path="./temp/customers.csv"
)

# Expected: Executes custom formula, saves results
```

### Test 3: Animated Visualization
```python
run_copilot(
    user_query="Create an animated scatter plot showing sales vs profit over years, with size representing quantity",
    csv_path="./temp/sales_history.csv"
)

# Expected: Creates animated Plotly HTML
```

---

## Impact

### Capabilities Unlocked

| Category | Before | After |
|----------|--------|-------|
| **Visualization Flexibility** | 11 predefined plot types | ∞ unlimited custom plots |
| **Custom Calculations** | Limited to built-in functions | ANY Python calculation |
| **Interactivity** | Basic Plotly plots | Full Plotly API (dropdowns, sliders, buttons) |
| **Data Transformations** | Predefined operations | ANY pandas/polars operation |
| **Export Formats** | CSV, Parquet | ANY format via Python |

### Real-World Examples

**Marketing Analytics:**
- "Create a cohort retention heatmap with custom color scales"
- "Build an interactive funnel analysis with drill-down by source"
- "Generate a CAC vs LTV scatter with industry benchmark lines"

**Sales Analytics:**
- "Plot quarterly revenue with comparison to same quarter last year, add trend line and forecast"
- "Create a geographical sales heatmap with state-level drill-down"
- "Build a sales pipeline dashboard with stage conversion metrics"

**Financial Analysis:**
- "Create a waterfall chart showing P&L breakdown by category"
- "Build an interactive cash flow projection with scenario sliders"
- "Generate a portfolio performance dashboard with asset allocation pie chart"

---

## Why This Matters

### The Philosophy

**Function Calling Bots** (limited):
```
Tool 1 → Task A
Tool 2 → Task B
Tool 3 → Task C
User Request D → ❌ "Sorry, I can't do that"
```

**True AI Agents** (unlimited):
```
Tool 1 → Task A
Tool 2 → Task B
Code Interpreter → Tasks C, D, E, F, G, H, ... ∞
User Request X → ✅ "Let me write code for that"
```

### From This Discussion

**Your Insight:**
> "This needs to be fixed so that if the user wants a specific plot we should be able to do it only then it will be a true AI Agent or else its just normal function calling"

**✅ FIXED:**
The agent can now handle **ANY data science request** by writing custom Python code when existing tools don't fit. This is what separates a true AI agent from a glorified function dispatcher.

---

## Next Steps

1. **Test with your bike sales dataset**:
   ```python
   run_copilot(
       user_query="Make a plot of extended sales by month for each bike model with a Plotly dropdown filter",
       csv_path="path/to/your/sales.csv"
   )
   ```

2. **Try other custom visualizations**:
   - "Create a 3D scatter plot with hover tooltips"
   - "Build a sunburst chart showing category hierarchy"
   - "Make an animated bar chart race over time"

3. **Push boundaries**:
   - "Generate a custom PDF report with matplotlib figures"
   - "Create an interactive map with geographical sales data"
   - "Build a real-time updating dashboard with Plotly Dash"

---

## Summary

✅ **Implemented**: Code Interpreter tool
✅ **Impact**: Transformed from function-calling bot to true AI agent
✅ **Capabilities**: Unlimited custom visualizations, calculations, transformations
✅ **Security**: Safe execution with timeout and dangerous operation blocking
✅ **Auto-imports**: pandas, numpy, matplotlib, seaborn, plotly ready to use
✅ **Smart guidance**: Agent knows WHEN to use code interpreter vs existing tools

**Your agent is now a TRUE AI AGENT** 🚀
