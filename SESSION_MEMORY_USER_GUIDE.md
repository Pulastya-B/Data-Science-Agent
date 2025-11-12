# Session Memory - User Guide

## Overview

Session memory enables the AI agent to remember previous interactions and resolve ambiguous follow-up requests. This makes conversations feel natural - you don't have to repeat dataset paths, target columns, or model names every time.

## Features

### 🧠 Context Tracking
- **Last Dataset**: Remembers which file you're working with
- **Last Model**: Remembers which model was trained (XGBoost, RandomForest, etc.)
- **Target Column**: Remembers prediction target
- **Best Score**: Remembers model performance
- **Output Files**: Tracks cleaned, encoded, and model files
- **Workflow History**: Complete record of all tool executions

### 🔗 Pronoun Resolution
Automatically resolves ambiguous references:
- "it" → last model or dataset
- "that" → last output file
- "the model" → last trained model
- "the data" → last dataset

### 💾 Persistence
- Sessions saved to SQLite database (`./cache_db/sessions.db`)
- Resume conversations after restarting the agent
- Automatic cleanup of old sessions (7 days default)

---

## Usage Examples

### Example 1: Cross-Validation Follow-Up

```python
from orchestrator import DataScienceCopilot

copilot = DataScienceCopilot(use_session_memory=True)

# Initial request - train model
copilot.analyze(
    file_path="./data/titanic.csv",
    task_description="Train a model to predict Survived",
    target_col="Survived"
)
# Agent trains XGBoost, achieves 0.85 accuracy
# Session stores: last_model="XGBoost", last_dataset="titanic.csv", last_target_col="Survived"

# Follow-up request (ambiguous!)
copilot.analyze(
    file_path="",  # EMPTY!
    task_description="Cross validate it",  # "it" = XGBoost
    target_col=None  # None!
)
# ✅ Agent automatically resolves:
#    - "it" → XGBoost (from session.last_model)
#    - file_path → "./data/titanic.csv" (from session.last_dataset)
#    - target_col → "Survived" (from session.last_target_col)
# → Runs cross-validation on XGBoost WITHOUT asking for clarification!
```

**What happens internally:**
```
User: "Cross validate it"
↓
session.resolve_ambiguity("Cross validate it")
↓
Returns: {
    "file_path": "./data/titanic.csv",
    "target_col": "Survived",
    "model_type": "xgboost"
}
↓
Agent calls: perform_cross_validation(
    file_path="./data/titanic.csv",
    target_col="Survived",
    model_type="xgboost"
)
```

---

### Example 2: Incremental Feature Engineering

```python
# Step 1: Clean data
copilot.analyze(
    file_path="./data/sales.csv",
    task_description="Clean the data"
)
# Session stores: last_output_files={"cleaned": "./outputs/data/cleaned.csv"}

# Step 2: Add features (no file specified!)
copilot.analyze(
    file_path="",  # Agent uses last output file
    task_description="Add time features to that"  # "that" = cleaned.csv
)
# ✅ Resolved: file_path = "./outputs/data/cleaned.csv" (from session)

# Step 3: Encode categories
copilot.analyze(
    file_path="",
    task_description="Encode it"  # "it" = last output with time features
)
# ✅ Resolved: file_path = "./outputs/data/time_features.csv"

# Step 4: Train model
copilot.analyze(
    file_path="",
    task_description="Train a model",
    target_col="revenue"
)
# ✅ Uses: "./outputs/data/encoded.csv" automatically
```

---

### Example 3: Model Comparison

```python
# Train first model
copilot.analyze(
    file_path="./data/house_prices.csv",
    task_description="Train XGBoost to predict SalePrice",
    target_col="SalePrice"
)
# Session: last_model="XGBoost", best_score=0.89

# Compare with different model (uses same dataset automatically)
copilot.analyze(
    file_path="",  # Automatically uses house_prices.csv
    task_description="Try that with Random Forest instead"
)
# ✅ Trains RandomForest on same dataset, same target
# Agent can compare: XGBoost (0.89) vs RandomForest (0.91) → RF wins!
```

---

### Example 4: Session Resumption

```python
# Day 1 - Morning
copilot1 = DataScienceCopilot(use_session_memory=True)
print(copilot1.get_session_id())  # "abc123-xyz789"

copilot1.analyze(
    file_path="./data/customers.csv",
    task_description="Profile and clean the dataset"
)
# Work continues...

# Day 1 - Afternoon (agent restarted)
copilot2 = DataScienceCopilot(
    use_session_memory=True,
    session_id="abc123-xyz789"  # Resume specific session
)

# Continue from where you left off
copilot2.analyze(
    file_path="",  # Uses customers.csv from session
    task_description="Train a model now"
)
# ✅ Agent remembers all previous work!
```

---

## API Reference

### Initialization

```python
DataScienceCopilot(
    use_session_memory=True,      # Enable session memory (default: True)
    session_id=None                # Resume specific session (None = auto-resume or create new)
)
```

**Auto-Resume Behavior:**
- If `session_id=None`, agent tries to resume most recent session (within 24 hours)
- If no recent session found, creates new one
- Session ID is auto-generated: UUID format

### Methods

#### `get_session_id() -> str`
Get current session ID.

```python
session_id = copilot.get_session_id()
print(f"Session: {session_id}")
```

#### `get_session_context() -> str`
Get human-readable session context summary.

```python
print(copilot.get_session_context())
# Output:
# **Session Context:**
# - Dataset: ./data/titanic.csv
# - Target Column: Survived
# - Last Model: XGBoost
# - Best Score: 0.8523
# - Task Type: classification
# - Output Files:
#   - cleaned: ./outputs/data/cleaned.csv
#   - encoded: ./outputs/data/encoded.csv
#   - model: ./outputs/models/xgboost_model.pkl
```

#### `clear_session()`
Clear current session context (start fresh).

```python
copilot.clear_session()
# Session context cleared, but session ID preserved
```

---

## Supported Ambiguities

Session memory automatically resolves these patterns:

### Cross-Validation
```python
"Cross validate it"
"Validate the model"
"Run CV on it"
```
→ Resolves: `file_path`, `target_col`, `model_type`

### Hyperparameter Tuning
```python
"Tune it"
"Optimize the model"
"Improve it with tuning"
```
→ Resolves: `file_path`, `target_col`, `model_type`

### Visualization
```python
"Plot it"
"Visualize the results"
"Show me graphs"
```
→ Resolves: `file_path`, `target_col`

### Feature Engineering
```python
"Add features to that"
"Create features on it"
"Engineer features"
```
→ Resolves: `file_path` (uses last output)

---

## Session Storage

### Database Location
```
./cache_db/sessions.db
```

### Schema
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_active TEXT NOT NULL,
    context_json TEXT NOT NULL
);
```

### Cleanup
Old sessions (>7 days) are automatically cleaned up.

Manual cleanup:
```python
from session_store import SessionStore

store = SessionStore()
deleted = store.cleanup_old_sessions(days=7)
print(f"Deleted {deleted} old sessions")
```

---

## Disabling Session Memory

If you want stateless behavior (each request independent):

```python
copilot = DataScienceCopilot(use_session_memory=False)
```

**Use case:** Batch processing, testing, or when explicit parameters are always provided.

---

## Testing

Run the test suite to verify session memory:

```bash
# Test all functionality
python test_session_memory.py --test all

# Test specific features
python test_session_memory.py --test memory      # Ambiguity resolution
python test_session_memory.py --test resumption  # Session resumption
python test_session_memory.py --test clear       # Session clearing
```

---

## Troubleshooting

### Issue: Agent doesn't remember context

**Symptom:** Agent asks "which dataset?" even after previous request

**Solution:**
1. Check session memory is enabled:
   ```python
   copilot = DataScienceCopilot(use_session_memory=True)  # ✅
   ```
2. Verify session is active:
   ```python
   print(copilot.get_session_context())
   ```
3. Ensure previous request completed successfully (context only updates on success)

### Issue: Session not resuming

**Symptom:** New session created instead of resuming previous one

**Causes:**
- Session older than 24 hours (default auto-resume window)
- Session ID not found in database
- Database file missing/corrupted

**Solution:**
```python
# Explicitly load session by ID
copilot = DataScienceCopilot(
    use_session_memory=True,
    session_id="your-session-id-here"
)
```

### Issue: Wrong context used

**Symptom:** Agent uses wrong dataset/model from session

**Solution:**
- Explicitly specify parameters when you want to override session:
  ```python
  copilot.analyze(
      file_path="./new_dataset.csv",  # Override session
      task_description="Train model",
      target_col="target"
  )
  ```
- Or clear session first:
  ```python
  copilot.clear_session()  # Start fresh
  ```

---

## Implementation Details

### Context Extraction

Session memory automatically extracts context from tool results:

| Tool | Extracts |
|------|----------|
| `profile_dataset` | `last_dataset` |
| `train_baseline_models` | `last_model`, `best_score`, `last_task_type`, `last_target_col` |
| `hyperparameter_tuning` | `best_score`, `last_model` |
| `clean_missing_values` | `last_output_files["cleaned"]` |
| `encode_categorical` | `last_output_files["encoded"]` |
| `auto_feature_engineering` | `last_output_files["engineered"]` |

### Model Name Normalization

Session memory normalizes model names for tool compatibility:

| Session Name | Normalized |
|--------------|------------|
| "XGBoost Classifier" | "xgboost" |
| "Random Forest" | "random_forest" |
| "Ridge Regression" | "ridge" |
| "Gradient Boosting" | "gradient_boosting" |

---

## Best Practices

### ✅ Do:
- Use session memory for interactive, multi-turn conversations
- Clear session when switching to completely different task
- Explicitly specify session_id when resuming important work
- Check session context before ambiguous requests

### ❌ Don't:
- Don't rely on session memory for critical production pipelines (use explicit parameters)
- Don't assume sessions persist forever (7-day default cleanup)
- Don't mix unrelated tasks in same session (clear between different datasets)

---

## Comparison: With vs Without Session Memory

### Without Session Memory ❌
```python
# Request 1
copilot.analyze(
    file_path="./data/sales.csv",
    task_description="Train XGBoost to predict revenue",
    target_col="revenue"
)

# Request 2 - Must repeat EVERYTHING
copilot.analyze(
    file_path="./data/sales.csv",        # Repeat
    task_description="Cross validate it",
    target_col="revenue"                 # Repeat
)
# Agent: "Error - specify model type!"

# Request 3 - Still must repeat
copilot.analyze(
    file_path="./data/sales.csv",        # Again...
    task_description="Tune XGBoost with 50 trials",
    target_col="revenue"                 # Again...
)
```

### With Session Memory ✅
```python
# Request 1
copilot.analyze(
    file_path="./data/sales.csv",
    task_description="Train XGBoost to predict revenue",
    target_col="revenue"
)

# Request 2 - Natural follow-up
copilot.analyze(
    file_path="",                    # Auto-filled!
    task_description="Cross validate it",
    target_col=None                  # Auto-filled!
)
# ✅ Works! Agent remembers: sales.csv, revenue, XGBoost

# Request 3 - Even simpler
copilot.analyze(
    file_path="",
    task_description="Tune it with 50 trials"
)
# ✅ Works! Agent has full context
```

**Result:**
- 50% less typing
- Faster iterations
- Natural conversation flow
- No repeated boilerplate

---

## Future Enhancements

Planned improvements:
- [ ] Multi-session management (switch between projects)
- [ ] Session branching (try alternative approaches)
- [ ] Session export/import (share sessions with team)
- [ ] Advanced context merging (combine insights from multiple sessions)
- [ ] Session analytics (track patterns across sessions)

---

## Conclusion

Session memory transforms the AI agent from a stateless tool into an intelligent assistant that remembers your work. It enables natural, multi-turn conversations without repetitive parameter specification.

**Key Benefits:**
- ✅ Natural follow-up commands ("cross validate it")
- ✅ Reduced verbosity (no repeated parameters)
- ✅ Context awareness (remembers what was done)
- ✅ Multi-turn workflows (clean → engineer → train → validate)
- ✅ Session persistence (resume after restart)

**Get Started:**
```python
from orchestrator import DataScienceCopilot

copilot = DataScienceCopilot(use_session_memory=True)

# First request
copilot.analyze(
    file_path="./data/my_data.csv",
    task_description="Train a model to predict target",
    target_col="target"
)

# Follow-ups are automatic!
copilot.analyze(file_path="", task_description="Cross validate it")
copilot.analyze(file_path="", task_description="Tune it")
copilot.analyze(file_path="", task_description="Plot the results")
```

🎉 **Session memory is now active by default!**
