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
from tools.tools_registry import TOOLS, get_all_tool_names, get_tools_by_category
from tools import (
    # Basic Tools (10)
    profile_dataset,
    detect_data_quality_issues,
    analyze_correlations,
    clean_missing_values,
    handle_outliers,
    fix_data_types,
    force_numeric_conversion,
    smart_type_inference,
    create_time_features,
    encode_categorical,
    train_baseline_models,
    generate_model_report,
    # Advanced Analysis (5)
    perform_eda_analysis,
    detect_model_issues,
    detect_anomalies,
    detect_and_handle_multicollinearity,
    perform_statistical_tests,
    # Advanced Feature Engineering (4)
    create_interaction_features,
    create_aggregation_features,
    engineer_text_features,
    auto_feature_engineering,
    # Advanced Preprocessing (3)
    handle_imbalanced_data,
    perform_feature_scaling,
    split_data_strategically,
    # Advanced Training (3)
    hyperparameter_tuning,
    train_ensemble_models,
    perform_cross_validation,
    # Business Intelligence (4)
    perform_cohort_analysis,
    perform_rfm_analysis,
    detect_causal_relationships,
    generate_business_insights,
    # Computer Vision (3)
    extract_image_features,
    perform_image_clustering,
    analyze_tabular_image_hybrid,
    # NLP/Text Analytics (4)
    perform_topic_modeling,
    perform_named_entity_recognition,
    analyze_sentiment_advanced,
    perform_text_similarity,
    # Production/MLOps (5)
    monitor_model_drift,
    explain_predictions,
    generate_model_card,
    perform_ab_test_analysis,
    detect_feature_leakage,
    # Time Series (3)
    forecast_time_series,
    detect_seasonality_trends,
    create_time_series_features,
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
        """Build mapping of tool names to their functions - All 46 tools."""
        return {
            # Basic Tools (10)
            "profile_dataset": profile_dataset,
            "detect_data_quality_issues": detect_data_quality_issues,
            "analyze_correlations": analyze_correlations,
            "clean_missing_values": clean_missing_values,
            "handle_outliers": handle_outliers,
            "fix_data_types": fix_data_types,
            "force_numeric_conversion": force_numeric_conversion,
            "smart_type_inference": smart_type_inference,
            "create_time_features": create_time_features,
            "encode_categorical": encode_categorical,
            "train_baseline_models": train_baseline_models,
            "generate_model_report": generate_model_report,
            # Advanced Analysis (5)
            "perform_eda_analysis": perform_eda_analysis,
            "detect_model_issues": detect_model_issues,
            "detect_anomalies": detect_anomalies,
            "detect_and_handle_multicollinearity": detect_and_handle_multicollinearity,
            "perform_statistical_tests": perform_statistical_tests,
            # Advanced Feature Engineering (4)
            "create_interaction_features": create_interaction_features,
            "create_aggregation_features": create_aggregation_features,
            "engineer_text_features": engineer_text_features,
            "auto_feature_engineering": auto_feature_engineering,
            # Advanced Preprocessing (3)
            "handle_imbalanced_data": handle_imbalanced_data,
            "perform_feature_scaling": perform_feature_scaling,
            "split_data_strategically": split_data_strategically,
            # Advanced Training (3)
            "hyperparameter_tuning": hyperparameter_tuning,
            "train_ensemble_models": train_ensemble_models,
            "perform_cross_validation": perform_cross_validation,
            # Business Intelligence (4)
            "perform_cohort_analysis": perform_cohort_analysis,
            "perform_rfm_analysis": perform_rfm_analysis,
            "detect_causal_relationships": detect_causal_relationships,
            "generate_business_insights": generate_business_insights,
            # Computer Vision (3)
            "extract_image_features": extract_image_features,
            "perform_image_clustering": perform_image_clustering,
            "analyze_tabular_image_hybrid": analyze_tabular_image_hybrid,
            # NLP/Text Analytics (4)
            "perform_topic_modeling": perform_topic_modeling,
            "perform_named_entity_recognition": perform_named_entity_recognition,
            "analyze_sentiment_advanced": analyze_sentiment_advanced,
            "perform_text_similarity": perform_text_similarity,
            # Production/MLOps (5)
            "monitor_model_drift": monitor_model_drift,
            "explain_predictions": explain_predictions,
            "generate_model_card": generate_model_card,
            "perform_ab_test_analysis": perform_ab_test_analysis,
            "detect_feature_leakage": detect_feature_leakage,
            # Time Series (3)
            "forecast_time_series": forecast_time_series,
            "detect_seasonality_trends": detect_seasonality_trends,
            "create_time_series_features": create_time_series_features,
        }
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt for the copilot."""
        return """You are an autonomous Data Science Agent. You EXECUTE tasks, not advise.

**CRITICAL: Complete the ENTIRE workflow. NEVER stop with recommendations.**

**WORKFLOW (Execute ALL steps):**
1. profile_dataset(file_path)
2. detect_data_quality_issues(file_path)
3. clean_missing_values(file_path, strategy="auto", output="./outputs/data/cleaned.csv")
4. handle_outliers(cleaned, method="clip", columns=["all"], output="./outputs/data/no_outliers.csv")
5. force_numeric_conversion(latest, columns=["all"], output="./outputs/data/numeric.csv", errors="coerce") ← CRITICAL for "no numeric features" errors
6. encode_categorical(latest, method="auto", output="./outputs/data/encoded.csv")
7. train_baseline_models(encoded, target_col, task_type="auto")
8. generate_model_report()

**KEY TOOLS (46 total available via function calling):**
- force_numeric_conversion: Converts string columns to numeric (auto-detects, skips text)
- clean_missing_values: "auto" mode supported
- encode_categorical: one-hot/target/frequency encoding
- train_baseline_models: Trains multiple models automatically
- Advanced: hyperparameter_tuning, train_ensemble_models, perform_eda_analysis, handle_imbalanced_data, perform_feature_scaling, detect_anomalies, detect_and_handle_multicollinearity, auto_feature_engineering, forecast_time_series, explain_predictions, generate_business_insights, perform_topic_modeling, extract_image_features, monitor_model_drift

**RULES:**
✅ EXECUTE each step (use tools)
✅ Use OUTPUT of each tool as INPUT to next
✅ If tool fails, continue pipeline
✅ If "no numeric features" → use force_numeric_conversion
✅ Save to ./outputs/data/
✅ When training fails → fix issue → RETRY
❌ NO recommendations without action
❌ NO stopping after detecting issues
❌ NO giving up on errors

File chain: original → cleaned.csv → no_outliers.csv → numeric.csv → encoded.csv → models

You are a DOER. Complete the ENTIRE pipeline automatically."""
    
    def _generate_cache_key(self, file_path: str, task_description: str, 
                           target_col: Optional[str] = None) -> str:
        """Generate cache key for a workflow."""
        # Include file hash to invalidate cache when data changes
        try:
            file_hash = self.cache.generate_file_hash(file_path)
        except:
            file_hash = "no_file"
        
        # Create simple string key (no kwargs unpacking to avoid dict hashing issues)
        cache_key_str = f"{file_hash}_{task_description}_{target_col or 'no_target'}"
        return self.cache._generate_key(cache_key_str)
    
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
    
    def _summarize_tool_result(self, tool_result: Dict[str, Any]) -> str:
        """
        Summarize tool result for LLM consumption.
        Extracts only essential info to avoid token bloat from large dataset outputs.
        """
        if not tool_result.get("success"):
            # Always return errors in full
            return json.dumps({
                "error": tool_result.get("error"),
                "error_type": tool_result.get("error_type")
            }, indent=2)
        
        result = tool_result.get("result", {})
        tool_name = tool_result.get("tool", "")
        
        # Create concise summary based on tool type
        summary = {"status": "success"}
        
        # Profile dataset - extract key stats only
        if tool_name == "profile_dataset":
            summary.update({
                "rows": result.get("basic_info", {}).get("num_rows"),
                "cols": result.get("basic_info", {}).get("num_columns"),
                "numeric_cols": len(result.get("numeric_columns", [])),
                "categorical_cols": len(result.get("categorical_columns", [])),
                "datetime_cols": len(result.get("datetime_columns", [])),
                "memory_mb": result.get("basic_info", {}).get("memory_usage_mb"),
                "missing_values": result.get("basic_info", {}).get("missing_values", 0)
            })
        
        # Data quality - extract issue counts
        elif tool_name == "detect_data_quality_issues":
            issues = result.get("issues", {})
            summary.update({
                "missing_values": len(issues.get("missing_values", [])),
                "duplicate_rows": result.get("duplicate_count", 0),
                "high_cardinality": len(issues.get("high_cardinality", [])),
                "constant_cols": len(issues.get("constant_columns", [])),
                "outliers": len(issues.get("outliers", [])),
                "total_issues": sum([
                    len(issues.get("missing_values", [])),
                    result.get("duplicate_count", 0),
                    len(issues.get("high_cardinality", [])),
                    len(issues.get("constant_columns", [])),
                    len(issues.get("outliers", []))
                ])
            })
        
        # File operations - just confirm path
        elif tool_name in ["clean_missing_values", "handle_outliers", "fix_data_types", 
                           "force_numeric_conversion", "encode_categorical", "smart_type_inference"]:
            summary.update({
                "output_path": result.get("output_path"),
                "message": result.get("message", ""),
                "rows_affected": result.get("rows_removed", result.get("rows_affected", 0))
            })
        
        # Training - extract model performance only
        elif tool_name == "train_baseline_models":
            models = result.get("models", {})
            best = result.get("best_model")
            summary.update({
                "best_model": best,
                "models_trained": list(models.keys()),
                "best_score": models.get(best, {}).get("test_metrics", {}).get("r2" if result.get("task_type") == "regression" else "accuracy"),
                "task_type": result.get("task_type")
            })
        
        # Report generation
        elif tool_name == "generate_model_report":
            summary.update({
                "report_path": result.get("report_path"),
                "message": "Report generated successfully"
            })
        
        # Default: extract message and status
        else:
            summary.update({
                "message": result.get("message", str(result)[:200]),  # Max 200 chars
                "output_path": result.get("output_path")
            })
        
        return json.dumps(summary, indent=2)
    
    def _format_tool_result(self, tool_result: Dict[str, Any]) -> str:
        """Format tool result for LLM consumption (alias for summarize)."""
        return self._summarize_tool_result(tool_result)
    
    def _compress_tools_registry(self) -> List[Dict]:
        """
        Create compressed version of tools registry.
        Keeps ALL 46 tools but removes verbose parameter descriptions.
        """
        compressed = []
        
        for tool in self.tools_registry:
            # Compress parameters by removing descriptions
            params = tool["function"]["parameters"]
            compressed_params = {
                "type": params["type"],
                "properties": {},
                "required": list(params.get("required", []))  # Create new list, not reference
            }
            
            # Keep only type info for properties, remove descriptions
            for prop_name, prop_value in params.get("properties", {}).items():
                compressed_prop = {}
                
                # Handle oneOf (like clean_missing_values strategy parameter)
                if "oneOf" in prop_value:
                    # Deep copy to avoid reference issues
                    compressed_prop["oneOf"] = json.loads(json.dumps(prop_value["oneOf"]))
                else:
                    compressed_prop["type"] = prop_value.get("type", "string")
                
                # Keep enum if present (important for validation)
                if "enum" in prop_value:
                    compressed_prop["enum"] = list(prop_value["enum"])  # Create new list
                
                # Keep array items type
                if prop_value.get("type") == "array" and "items" in prop_value:
                    compressed_prop["items"] = {"type": prop_value["items"].get("type", "string")}
                
                compressed_params["properties"][prop_name] = compressed_prop
            
            compressed_tool = {
                "type": tool["type"],
                "function": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"][:100],  # Short description
                    "parameters": compressed_params
                }
            }
            compressed.append(compressed_tool)
        
        return compressed
    
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

Execute the complete workflow: profile → clean → convert types → encode → train → report."""
        
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
                # Prune messages to avoid token bloat (keep system + user + last 8 messages)
                if len(messages) > 10:
                    # Keep: system prompt, user message, and last 8 tool interactions
                    messages = [messages[0], messages[1]] + messages[-8:]
                    print(f"📊 Pruned conversation history (keeping last 8 messages)")
                
                # Use compressed tools registry (all 46 tools but shorter descriptions)
                tools_to_use = self._compress_tools_registry()
                
                # Call Groq with function calling
                response = self.groq_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools_to_use,
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
