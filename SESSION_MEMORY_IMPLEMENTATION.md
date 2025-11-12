# Session-Based Memory Implementation Plan

## Problem Statement

**Current Issue:**
- Agent has NO memory between requests
- Cannot handle follow-up commands like:
  - "cross validate it" (which model?)
  - "add this feature" (which dataset?)
  - "try that with Ridge instead" (which task?)
  - "plot the results" (which results?)

**Example Failure Scenario:**
```
User: "Train a model on earthquake.csv to predict magnitude"
Agent: ✅ Trains XGBoost, achieves 0.92 accuracy

User: "Cross validate it"
Agent: ❌ "Cross validate what? Please specify the model and dataset"
          (Should remember: XGBoost on earthquake.csv with target='mag')
```

## Requirements

### 1. **Session Context Storage**
Store per-session state:
```python
{
    "session_id": "abc123",
    "last_dataset": "./data/earthquake.csv",
    "last_target_col": "mag",
    "last_model": "XGBoost",
    "last_task_type": "regression",
    "best_score": 0.92,
    "workflow_history": [...],
    "last_output_files": {
        "cleaned": "./outputs/data/cleaned.csv",
        "encoded": "./outputs/data/encoded.csv",
        "model": "./outputs/models/xgboost_model.pkl"
    },
    "conversation_context": [
        {"user": "Train model", "agent": "Trained XGBoost, 0.92 score"}
    ]
}
```

### 2. **Context-Aware Intent Resolution**
Resolve ambiguous requests using session memory:
```python
User: "cross validate it"
↓
session.get("last_model") → "XGBoost"
session.get("last_dataset") → "./data/earthquake.csv"
session.get("last_target_col") → "mag"
↓
Resolved: perform_cross_validation(
    file_path="./data/earthquake.csv",
    target_col="mag",
    model_type="xgboost"
)
```

### 3. **Pronoun Resolution**
Handle pronouns and references:
- "it" → last model/dataset
- "that" → last output
- "the model" → last trained model
- "add this feature" → to last dataset

## Implementation Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DataScienceCopilot                      │
├─────────────────────────────────────────────────────────────┤
│  - analyze()                                                │
│  - _build_system_prompt()                                  │
│  - _execute_tool()                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │    SessionMemory           │
            ├────────────────────────────┤
            │  - session_id              │
            │  - last_dataset            │
            │  - last_target_col         │
            │  - last_model              │
            │  - workflow_history        │
            │  - conversation_context    │
            │                            │
            │  Methods:                  │
            │  - update()                │
            │  - get_context()           │
            │  - resolve_ambiguity()     │
            │  - add_conversation()      │
            │  - clear()                 │
            └────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │  SessionStore (SQLite)     │
            ├────────────────────────────┤
            │  sessions.db               │
            │                            │
            │  Table: sessions           │
            │  - session_id (PK)         │
            │  - created_at              │
            │  - last_active             │
            │  - context_json            │
            └────────────────────────────┘
```

### File Structure

```
src/
├── orchestrator.py (modified - add session support)
├── session_memory.py (NEW)
└── session_store.py (NEW)

cache_db/
└── sessions.db (NEW - SQLite database)
```

## Implementation Steps

### Step 1: Create SessionMemory Class

**File:** `src/session_memory.py`

```python
"""
Session Memory Manager
Maintains context across user interactions for intelligent follow-up handling.
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

class SessionMemory:
    """
    Manages session-based memory for contextual AI interactions.
    
    Features:
    - Stores last dataset, model, target column
    - Tracks workflow history
    - Resolves ambiguous pronouns ("it", "that", "the model")
    - Maintains conversation context
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize session memory.
        
        Args:
            session_id: Unique session identifier (auto-generated if None)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        
        # Core context
        self.last_dataset: Optional[str] = None
        self.last_target_col: Optional[str] = None
        self.last_model: Optional[str] = None
        self.last_task_type: Optional[str] = None
        self.best_score: Optional[float] = None
        
        # Output tracking
        self.last_output_files: Dict[str, str] = {}
        
        # Workflow history
        self.workflow_history: List[Dict[str, Any]] = []
        
        # Conversation context (for pronoun resolution)
        self.conversation_context: List[Dict[str, str]] = []
        
        # Tool results cache
        self.last_tool_results: Dict[str, Any] = {}
    
    def update(self, **kwargs):
        """
        Update session context with new information.
        
        Args:
            last_dataset: Path to dataset
            last_target_col: Target column name
            last_model: Model name (XGBoost, RandomForest, etc.)
            last_task_type: Task type (regression, classification)
            best_score: Best model score
            last_output_files: Dict of output file paths
        """
        self.last_active = datetime.now()
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def add_workflow_step(self, tool_name: str, result: Dict[str, Any]):
        """Add a workflow step to history."""
        self.workflow_history.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "result": result
        })
        
        # Update context based on tool results
        self._extract_context_from_tool(tool_name, result)
    
    def _extract_context_from_tool(self, tool_name: str, result: Dict[str, Any]):
        """Extract relevant context from tool execution."""
        
        # Track dataset from profiling
        if tool_name == "profile_dataset" and "file_path" in result:
            self.last_dataset = result.get("file_path")
        
        # Track model training results
        if tool_name == "train_baseline_models":
            best_model = result.get("best_model", {})
            if isinstance(best_model, dict):
                self.last_model = best_model.get("name")
                self.best_score = best_model.get("score")
            else:
                self.last_model = best_model
            
            self.last_task_type = result.get("task_type")
            self.last_target_col = result.get("target_col")
        
        # Track output files
        if "output_path" in result:
            tool_category = self._categorize_tool(tool_name)
            self.last_output_files[tool_category] = result["output_path"]
        
        # Store tool results
        self.last_tool_results[tool_name] = result
    
    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize tool for output tracking."""
        if "clean" in tool_name:
            return "cleaned"
        elif "encode" in tool_name:
            return "encoded"
        elif "train" in tool_name or "model" in tool_name:
            return "model"
        elif "plot" in tool_name or "visual" in tool_name:
            return "visualization"
        else:
            return "other"
    
    def add_conversation(self, user_message: str, agent_response: str):
        """Add conversation turn to context."""
        self.conversation_context.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "agent": agent_response
        })
        
        # Keep only last 10 turns
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
    
    def resolve_ambiguity(self, task_description: str) -> Dict[str, Any]:
        """
        Resolve ambiguous references in user request.
        
        Args:
            task_description: User's request (may contain "it", "that", etc.)
        
        Returns:
            Dict with resolved parameters
        """
        task_lower = task_description.lower()
        resolved = {}
        
        # Pronouns that reference last model/dataset
        ambiguous_refs = ["it", "that", "this", "the model", "the dataset"]
        has_ambiguous_ref = any(ref in task_lower for ref in ambiguous_refs)
        
        # Cross-validation requests
        if "cross validat" in task_lower or "cv" in task_lower:
            if has_ambiguous_ref or not any(word in task_lower for word in ["model", "dataset", "target"]):
                resolved.update({
                    "file_path": self.last_output_files.get("encoded") or self.last_dataset,
                    "target_col": self.last_target_col,
                    "model_type": self._normalize_model_name(self.last_model)
                })
        
        # Hyperparameter tuning requests
        if "tun" in task_lower or "optim" in task_lower:
            if has_ambiguous_ref:
                resolved.update({
                    "file_path": self.last_output_files.get("encoded") or self.last_dataset,
                    "target_col": self.last_target_col,
                    "model_type": self._normalize_model_name(self.last_model)
                })
        
        # Visualization requests referencing "the results"
        if ("plot" in task_lower or "visualiz" in task_lower) and ("result" in task_lower or "it" in task_lower):
            resolved.update({
                "file_path": self.last_dataset
            })
        
        # "Add feature" requests
        if "add feature" in task_lower or "create feature" in task_lower:
            if has_ambiguous_ref:
                resolved.update({
                    "file_path": self.last_output_files.get("encoded") or self.last_dataset
                })
        
        return resolved
    
    def _normalize_model_name(self, model_name: Optional[str]) -> Optional[str]:
        """Normalize model name for tool compatibility."""
        if not model_name:
            return None
        
        name_lower = model_name.lower()
        
        if "xgb" in name_lower:
            return "xgboost"
        elif "random" in name_lower or "forest" in name_lower:
            return "random_forest"
        elif "ridge" in name_lower:
            return "ridge"
        elif "lasso" in name_lower:
            return "ridge"  # Use ridge for lasso
        elif "logistic" in name_lower:
            return "logistic"
        else:
            return model_name
    
    def get_context_summary(self) -> str:
        """Generate human-readable context summary."""
        if not self.last_dataset:
            return "No previous context available."
        
        summary = f"**Session Context:**\n"
        summary += f"- Dataset: {self.last_dataset}\n"
        
        if self.last_target_col:
            summary += f"- Target Column: {self.last_target_col}\n"
        
        if self.last_model:
            summary += f"- Last Model: {self.last_model}\n"
        
        if self.best_score is not None:
            summary += f"- Best Score: {self.best_score:.4f}\n"
        
        if self.last_task_type:
            summary += f"- Task Type: {self.last_task_type}\n"
        
        if self.last_output_files:
            summary += f"- Output Files:\n"
            for category, path in self.last_output_files.items():
                summary += f"  - {category}: {path}\n"
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "last_dataset": self.last_dataset,
            "last_target_col": self.last_target_col,
            "last_model": self.last_model,
            "last_task_type": self.last_task_type,
            "best_score": self.best_score,
            "last_output_files": self.last_output_files,
            "workflow_history": self.workflow_history,
            "conversation_context": self.conversation_context,
            "last_tool_results": self.last_tool_results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMemory':
        """Deserialize session from dictionary."""
        session = cls(session_id=data.get("session_id"))
        
        session.created_at = datetime.fromisoformat(data.get("created_at"))
        session.last_active = datetime.fromisoformat(data.get("last_active"))
        session.last_dataset = data.get("last_dataset")
        session.last_target_col = data.get("last_target_col")
        session.last_model = data.get("last_model")
        session.last_task_type = data.get("last_task_type")
        session.best_score = data.get("best_score")
        session.last_output_files = data.get("last_output_files", {})
        session.workflow_history = data.get("workflow_history", [])
        session.conversation_context = data.get("conversation_context", [])
        session.last_tool_results = data.get("last_tool_results", {})
        
        return session
    
    def clear(self):
        """Clear all session context."""
        self.last_dataset = None
        self.last_target_col = None
        self.last_model = None
        self.last_task_type = None
        self.best_score = None
        self.last_output_files = {}
        self.workflow_history = []
        self.conversation_context = []
        self.last_tool_results = {}
```

### Step 2: Create SessionStore for Persistence

**File:** `src/session_store.py`

```python
"""
Session Storage Manager
Persists session memory to SQLite database.
"""

import sqlite3
import json
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta

from session_memory import SessionMemory

class SessionStore:
    """
    Persistent storage for session memory using SQLite.
    """
    
    def __init__(self, db_path: str = "./cache_db/sessions.db"):
        """
        Initialize session store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create sessions table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL,
                context_json TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save(self, session: SessionMemory):
        """Save session to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data = session.to_dict()
        
        cursor.execute("""
            INSERT OR REPLACE INTO sessions (session_id, created_at, last_active, context_json)
            VALUES (?, ?, ?, ?)
        """, (
            session.session_id,
            session.created_at.isoformat(),
            session.last_active.isoformat(),
            json.dumps(data)
        ))
        
        conn.commit()
        conn.close()
    
    def load(self, session_id: str) -> Optional[SessionMemory]:
        """Load session from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT context_json FROM sessions WHERE session_id = ?
        """, (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        data = json.loads(result[0])
        return SessionMemory.from_dict(data)
    
    def get_recent_session(self, max_age_hours: int = 24) -> Optional[SessionMemory]:
        """Get most recent active session within time window."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        
        cursor.execute("""
            SELECT context_json FROM sessions
            WHERE last_active > ?
            ORDER BY last_active DESC
            LIMIT 1
        """, (cutoff_time,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        data = json.loads(result[0])
        return SessionMemory.from_dict(data)
    
    def delete(self, session_id: str):
        """Delete session from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
    
    def cleanup_old_sessions(self, days: int = 7):
        """Delete sessions older than specified days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("DELETE FROM sessions WHERE last_active < ?", (cutoff_time,))
        
        conn.commit()
        conn.close()
```

### Step 3: Integrate into DataScienceCopilot

**Modifications to `src/orchestrator.py`:**

```python
# Add imports at top
from session_memory import SessionMemory
from session_store import SessionStore

class DataScienceCopilot:
    def __init__(self, ..., session_id: Optional[str] = None, use_session_memory: bool = True):
        # ... existing init code ...
        
        # Initialize session memory
        self.use_session_memory = use_session_memory
        if use_session_memory:
            self.session_store = SessionStore()
            
            # Try to load existing session or create new one
            if session_id:
                self.session = self.session_store.load(session_id)
                if not self.session:
                    self.session = SessionMemory(session_id=session_id)
            else:
                # Try to continue recent session
                self.session = self.session_store.get_recent_session(max_age_hours=24)
                if not self.session:
                    self.session = SessionMemory()
            
            print(f"📝 Session: {self.session.session_id}")
            if self.session.last_dataset:
                print(f"   Continuing from: {self.session.last_dataset}")
        else:
            self.session = None
    
    def analyze(self, file_path: str, task_description: str, ...):
        # ... existing code ...
        
        # 🧠 RESOLVE AMBIGUITY USING SESSION MEMORY
        if self.session:
            resolved_params = self.session.resolve_ambiguity(task_description)
            
            # Use resolved params if user didn't specify
            if not file_path and resolved_params.get("file_path"):
                file_path = resolved_params["file_path"]
                print(f"📝 Using dataset from session: {file_path}")
            
            if not target_col and resolved_params.get("target_col"):
                target_col = resolved_params["target_col"]
                print(f"📝 Using target column from session: {target_col}")
        
        # ... rest of existing code ...
        
        # After workflow completes, update session
        if self.session:
            self.session.add_conversation(task_description, final_summary)
            self.session_store.save(self.session)
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = # ... existing tool execution ...
        
        # Update session memory after each tool
        if self.session:
            self.session.add_workflow_step(tool_name, result)
        
        return result
```

## Usage Examples

### Example 1: Cross-Validation Follow-Up

```python
# First request
copilot.analyze(
    file_path="./data/earthquake.csv",
    task_description="Train a model to predict magnitude",
    target_col="mag"
)
# Agent trains XGBoost, stores: last_model="XGBoost", last_dataset="earthquake.csv", last_target_col="mag"

# Follow-up request (ambiguous)
copilot.analyze(
    file_path="",  # Empty!
    task_description="Cross validate it",  # "it" = XGBoost
    target_col=None
)
# Agent resolves:
#   - "it" → XGBoost (from session.last_model)
#   - file_path → "./data/earthquake.csv" (from session.last_dataset)
#   - target_col → "mag" (from session.last_target_col)
# → Calls: perform_cross_validation(file_path="earthquake.csv", target_col="mag", model_type="xgboost")
```

### Example 2: Incremental Feature Engineering

```python
# First request
copilot.analyze(
    file_path="./data/sales.csv",
    task_description="Clean the data and encode categories"
)
# Agent cleans, encodes, stores: last_output_files={"encoded": "encoded.csv"}

# Follow-up (no file specified)
copilot.analyze(
    file_path="",
    task_description="Add time features to that"  # "that" = encoded.csv
)
# Agent resolves:
#   - "that" → "./outputs/data/encoded.csv" (from session.last_output_files)
# → Calls: create_time_features(file_path="encoded.csv", ...)
```

### Example 3: Model Comparison

```python
# First request
copilot.analyze(
    file_path="./data/titanic.csv",
    task_description="Train XGBoost to predict Survived"
)
# Stores: last_model="XGBoost", best_score=0.85

# Compare with different model
copilot.analyze(
    file_path="",
    task_description="Try that with Random Forest instead"
)
# Agent resolves:
#   - "that" → same dataset (./data/titanic.csv)
#   - target_col → "Survived"
# → Trains RandomForest, compares with 0.85 baseline
```

## Benefits

✅ **Natural Conversations:**
- Users can say "cross validate it" instead of repeating all parameters
- Feels like talking to a human assistant

✅ **Reduced Verbosity:**
- No need to repeat dataset path, target column every time
- Faster iterations

✅ **Context Awareness:**
- Agent remembers what was done previously
- Can make intelligent suggestions based on history

✅ **Multi-Turn Workflows:**
- "Clean data" → "Add features" → "Train model" → "Cross validate it"
- Each step builds on the previous

✅ **Error Recovery:**
- If something fails, user can say "try that again with different parameters"
- Agent knows which task to retry

## Next Steps

1. Create `src/session_memory.py` ✅ (code provided above)
2. Create `src/session_store.py` ✅ (code provided above)
3. Modify `src/orchestrator.py` to integrate session memory
4. Add session management to system prompt (teach LLM about context)
5. Test with multi-turn conversations
6. Add session cleanup/reset commands

## Testing Scenarios

### Test 1: Cross-Validation Follow-Up
```
User: "Train model on earthquake.csv predicting mag"
Agent: [Trains XGBoost, 0.92 score]

User: "Cross validate it"
Expected: ✅ Runs CV on XGBoost (should NOT ask "which model?")
```

### Test 2: Pronoun Resolution
```
User: "Clean titanic.csv"
Agent: [Cleans data → cleaned.csv]

User: "Add time features to that"
Expected: ✅ Uses cleaned.csv (should NOT ask "which file?")
```

### Test 3: Incremental Workflow
```
User: "Load sales.csv"
User: "Clean it"
User: "Encode categories"
User: "Train a model"
Expected: ✅ Each step uses previous output automatically
```

## Priority: HIGH

This is a critical missing feature that significantly impacts user experience and makes the agent feel less intelligent.

**Recommendation:** Implement ASAP before adding more tools/features.
