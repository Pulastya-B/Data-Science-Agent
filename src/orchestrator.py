"""
Data Science Copilot Orchestrator
Main orchestration class that uses Groq's function calling to execute data science workflows.
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import time

from groq import Groq
from dotenv import load_dotenv

from cache.cache_manager import CacheManager
from tools.tools_registry import TOOLS, get_all_tool_names
from tools import (
    profile_dataset,
    detect_data_quality_issues,
    analyze_correlations,
    clean_missing_values,
    handle_outliers,
    fix_data_types,
    create_time_features,
    encode_categorical,
    train_baseline_models,
    generate_model_report,
)


class DataScienceCopilot:
    """
    Main orchestrator for data science workflows using Groq's GPT-OSS-120B.
    
    Uses function calling to intelligently route to data profiling, cleaning,
    feature engineering, and model training tools.
    """
    
    def __init__(self, groq_api_key: Optional[str] = None, 
                 cache_db_path: Optional[str] = None,
                 reasoning_effort: str = "medium"):
        """
        Initialize the Data Science Copilot.
        
        Args:
            groq_api_key: Groq API key (or set GROQ_API_KEY env var)
            cache_db_path: Path to cache database
            reasoning_effort: Reasoning effort for Groq ('low', 'medium', 'high')
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize Groq client
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key must be provided or set in GROQ_API_KEY env var")
        
        self.groq_client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.reasoning_effort = reasoning_effort
        
        # Initialize cache
        cache_path = cache_db_path or os.getenv("CACHE_DB_PATH", "./cache_db/cache.db")
        self.cache = CacheManager(db_path=cache_path)
        
        # Tools registry
        self.tools_registry = TOOLS
        self.tool_functions = self._build_tool_functions_map()
        
        # Token tracking
        self.total_tokens_used = 0
        self.api_calls_made = 0
        
        # Ensure output directories exist
        Path("./outputs").mkdir(exist_ok=True)
        Path("./outputs/models").mkdir(exist_ok=True)
        Path("./outputs/reports").mkdir(exist_ok=True)
        Path("./outputs/data").mkdir(exist_ok=True)
    
    def _build_tool_functions_map(self) -> Dict[str, callable]:
        """Build mapping of tool names to their functions."""
        return {
            "profile_dataset": profile_dataset,
            "detect_data_quality_issues": detect_data_quality_issues,
            "analyze_correlations": analyze_correlations,
            "clean_missing_values": clean_missing_values,
            "handle_outliers": handle_outliers,
            "fix_data_types": fix_data_types,
            "create_time_features": create_time_features,
            "encode_categorical": encode_categorical,
            "train_baseline_models": train_baseline_models,
            "generate_model_report": generate_model_report,
        }
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt for the copilot."""
        return """You are an autonomous Data Science Agent. You EXECUTE data science tasks, not just provide advice.

**CRITICAL: You must COMPLETE the entire workflow, not just detect issues.**

**Your Tools:**
1. profile_dataset - Get dataset overview
2. detect_data_quality_issues - Find problems
3. analyze_correlations - Find feature relationships
4. clean_missing_values - Fill/drop missing data
5. handle_outliers - Remove or cap outliers
6. fix_data_types - Convert column types
7. create_time_features - Extract date features
8. encode_categorical - Convert categorical to numeric
9. train_baseline_models - Train ML models
10. generate_model_report - Create evaluation report

**EXECUTION WORKFLOW (Complete ALL steps):**

1. **Profile** → Use `profile_dataset(file_path)`
2. **Detect Issues** → Use `detect_data_quality_issues(file_path)`
3. **Clean Missing** → ALWAYS use `clean_missing_values(file_path, strategy="auto", output_path="./outputs/data/cleaned.csv")`
4. **Handle Outliers** → If outliers found, use `handle_outliers(cleaned_path, method="clip", output_path="./outputs/data/no_outliers.csv")`
5. **Encode Categorical** → ALWAYS use `encode_categorical(latest_path, method="auto", output_path="./outputs/data/encoded.csv")`
6. **Train Models** → Use `train_baseline_models(encoded_path, target_col, task_type="auto")`
7. **Generate Report** → Use `generate_model_report()`

**CRITICAL RULES:**
- ✅ EXECUTE each step, don't just recommend
- ✅ Save intermediate files to ./outputs/data/ with descriptive names
- ✅ Use the OUTPUT of each tool as INPUT to the next
- ✅ If a tool fails, adapt and continue (skip optional steps)
- ✅ ALWAYS clean, encode, and train - complete the full pipeline
- ❌ DO NOT just list recommendations
- ❌ DO NOT stop after detecting issues
- ❌ DO NOT give advice without taking action

**File Path Chain:**
Original → cleaned.csv → no_outliers.csv → encoded.csv → train_baseline_models()

**When user asks to "train a model":**
1. Profile the data
2. Clean ALL issues found
3. Encode categorical variables
4. Train the models
5. Report results

You are a DOER, not an advisor. Complete the entire pipeline automatically."""
    
    def _generate_cache_key(self, file_path: str, task_description: str, 
                           target_col: Optional[str] = None) -> str:
        """Generate cache key for a workflow."""
        # Include file hash to invalidate cache when data changes
        try:
            file_hash = self.cache.generate_file_hash(file_path)
        except:
            file_hash = "no_file"
        
        cache_data = {
            "file_hash": file_hash,
            "task": task_description,
            "target": target_col
        }
        
        return self.cache._generate_key(**cache_data)
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single tool function.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tool_functions:
            return {
                "error": f"Tool '{tool_name}' not found",
                "available_tools": get_all_tool_names()
            }
        
        try:
            tool_func = self.tool_functions[tool_name]
            result = tool_func(**arguments)
            return {
                "success": True,
                "tool": tool_name,
                "result": result
            }
        
        except Exception as e:
            return {
                "success": False,
                "tool": tool_name,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def _format_tool_result(self, tool_result: Dict[str, Any]) -> str:
        """Format tool result for LLM consumption."""
        if tool_result.get("success"):
            return json.dumps(tool_result["result"], indent=2)
        else:
            return json.dumps({
                "error": tool_result.get("error"),
                "error_type": tool_result.get("error_type")
            }, indent=2)
    
    def analyze(self, file_path: str, task_description: str, 
               target_col: Optional[str] = None, 
               use_cache: bool = True,
               stream: bool = True,
               max_iterations: int = 10) -> Dict[str, Any]:
        """
        Main entry point for data science analysis.
        
        Args:
            file_path: Path to dataset file
            task_description: Natural language description of the task
            target_col: Optional target column name
            use_cache: Whether to use cached results
            stream: Whether to stream LLM responses
            max_iterations: Maximum number of tool execution iterations
            
        Returns:
            Analysis results including summary and tool outputs
        """
        start_time = time.time()
        
        # Check cache
        if use_cache:
            cache_key = self._generate_cache_key(file_path, task_description, target_col)
            cached = self.cache.get(cache_key)
            if cached:
                print("✓ Using cached results")
                return cached
        
        # Build initial messages
        system_prompt = self._build_system_prompt()
        
        user_message = f"""Please analyze the dataset and complete the following task:

**Dataset**: {file_path}
**Task**: {task_description}
**Target Column**: {target_col if target_col else 'Not specified - please infer from data'}

Please execute the complete data science workflow:
1. Profile the dataset
2. Detect and fix data quality issues
3. Engineer relevant features
4. Train baseline models
5. Generate evaluation report

Provide a comprehensive analysis with clear recommendations."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Track workflow
        workflow_history = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                # Call Groq with function calling
                response = self.groq_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools_registry,
                    tool_choice="auto",
                    temperature=0.1,  # Low temperature for consistent outputs
                    max_tokens=4096
                )
                
                self.api_calls_made += 1
                
                # Get response message
                response_message = response.choices[0].message
                
                # Check if done (no tool calls)
                if not response_message.tool_calls:
                    # Final response
                    final_summary = response_message.content
                    
                    result = {
                        "status": "success",
                        "summary": final_summary,
                        "workflow_history": workflow_history,
                        "iterations": iteration,
                        "api_calls": self.api_calls_made,
                        "execution_time": round(time.time() - start_time, 2)
                    }
                    
                    # Cache result
                    if use_cache:
                        self.cache.set(cache_key, result, metadata={
                            "file_path": file_path,
                            "task": task_description
                        })
                    
                    return result
                
                # Execute tool calls
                messages.append(response_message)
                
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"\n🔧 Executing: {tool_name}")
                    print(f"   Arguments: {json.dumps(tool_args, indent=2)}")
                    
                    # Execute tool
                    tool_result = self._execute_tool(tool_name, tool_args)
                    
                    # Track in workflow
                    workflow_history.append({
                        "iteration": iteration,
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": tool_result
                    })
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": self._format_tool_result(tool_result)
                    })
                    
                    print(f"   ✓ Completed: {tool_name}")
            
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "workflow_history": workflow_history,
                    "iterations": iteration
                }
        
        # Max iterations reached
        return {
            "status": "incomplete",
            "message": f"Reached maximum iterations ({max_iterations})",
            "workflow_history": workflow_history,
            "iterations": iteration
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache.clear_all()
