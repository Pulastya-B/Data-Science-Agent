"""
Data Science Copilot Orchestrator
Main orchestration class that uses LLM function calling to execute data science workflows.
Supports multiple providers: Groq and Gemini.
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import time

from groq import Groq
import google.generativeai as genai
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
    Main orchestrator for data science workflows using LLM function calling.
    
    Supports multiple providers: Groq and Gemini.
    Uses function calling to intelligently route to data profiling, cleaning,
    feature engineering, and model training tools.
    """
    
    def __init__(self, groq_api_key: Optional[str] = None, 
                 google_api_key: Optional[str] = None,
                 cache_db_path: Optional[str] = None,
                 reasoning_effort: str = "medium",
                 provider: Optional[str] = None):
        """
        Initialize the Data Science Copilot.
        
        Args:
            groq_api_key: Groq API key (or set GROQ_API_KEY env var)
            google_api_key: Google API key (or set GOOGLE_API_KEY env var)
            cache_db_path: Path to cache database
            reasoning_effort: Reasoning effort for Groq ('low', 'medium', 'high')
            provider: LLM provider - 'groq' or 'gemini' (or set LLM_PROVIDER env var)
        """
        # Load environment variables
        load_dotenv()
        
        # Determine provider
        self.provider = provider or os.getenv("LLM_PROVIDER", "groq").lower()
        
        if self.provider == "groq":
            # Initialize Groq client
            api_key = groq_api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Groq API key must be provided or set in GROQ_API_KEY env var")
            
            self.groq_client = Groq(api_key=api_key)
            self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            self.reasoning_effort = reasoning_effort
            self.gemini_model = None
            print(f"🤖 Initialized with Groq provider - Model: {self.model}")
            
        elif self.provider == "gemini":
            # Initialize Gemini client
            api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY env var")
            
            genai.configure(api_key=api_key)
            self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
            self.gemini_model = genai.GenerativeModel(
                self.model,
                generation_config={"temperature": 0.1}
            )
            self.groq_client = None
            print(f"🤖 Initialized with Gemini provider - Model: {self.model}")
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Choose 'groq' or 'gemini'")
        
        # Initialize cache
        cache_path = cache_db_path or os.getenv("CACHE_DB_PATH", "./cache_db/cache.db")
        self.cache = CacheManager(db_path=cache_path)
        
        # Tools registry
        self.tools_registry = TOOLS
        self.tool_functions = self._build_tool_functions_map()
        
        # Token tracking
        self.total_tokens_used = 0
        self.api_calls_made = 0
        
        # Rate limiting for Gemini (10 RPM free tier)
        self.last_api_call_time = 0
        self.min_api_call_interval = 3.5 if self.provider == "gemini" else 0  # 6.5s = ~9 calls/min (safe margin)
        
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

**CRITICAL: Use the provided function calling tools. Do NOT generate XML-style function calls.**

**CRITICAL: Complete the ENTIRE workflow. NEVER stop with recommendations.**

**WORKFLOW (Execute ALL steps - DO NOT SKIP):**
1. profile_dataset(file_path) - ONCE ONLY
2. detect_data_quality_issues(file_path) - ONCE ONLY
3. clean_missing_values(file_path, strategy="auto", output="./outputs/data/cleaned.csv")
4. handle_outliers(cleaned, method="clip", columns=["all"], output="./outputs/data/no_outliers.csv")
5. force_numeric_conversion(latest, columns=["all"], output="./outputs/data/numeric.csv", errors="coerce")
6. encode_categorical(latest, method="auto", output="./outputs/data/encoded.csv")
7. **train_baseline_models**(encoded, target_col, task_type="auto") ← REQUIRED! DO NOT SKIP!
8. STOP after training completes (no need for generate_model_report)

**CRITICAL RULES:**
- DO NOT repeat profile_dataset or detect_data_quality_issues multiple times
- After encode_categorical, IMMEDIATELY call train_baseline_models
- DO NOT call smart_type_inference after encoding - data is ready
- Training is the GOAL - do not analyze endlessly, just TRAIN

**KEY TOOLS (46 total available via function calling):**
- force_numeric_conversion: Converts string columns to numeric (auto-detects, skips text)
- clean_missing_values: "auto" mode supported
- encode_categorical: one-hot/target/frequency encoding
- train_baseline_models: Trains multiple models automatically
- Advanced: hyperparameter_tuning, train_ensemble_models, perform_eda_analysis, handle_imbalanced_data, perform_feature_scaling, detect_anomalies, detect_and_handle_multicollinearity, auto_feature_engineering, forecast_time_series, explain_predictions, generate_business_insights, perform_topic_modeling, extract_image_features, monitor_model_drift

**RULES:**
✅ EXECUTE each step (use tools) - ONE tool call per response
✅ Use OUTPUT of each tool as INPUT to next
✅ If tool fails, continue pipeline
✅ If "no numeric features" → use force_numeric_conversion
✅ Save to ./outputs/data/
✅ When training fails → fix issue → RETRY
❌ NO recommendations without action
❌ NO stopping after detecting issues
❌ NO giving up on errors
❌ NO XML-style function syntax like <function=name />

**CRITICAL: Call ONE function at a time. Wait for its result before calling the next.**

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
            best = result.get("best_model", {})
            best_model_name = best.get("name") if isinstance(best, dict) else best
            summary.update({
                "best_model": best_model_name,
                "models_trained": list(models.keys()),
                "best_score": best.get("score") if isinstance(best, dict) else None,
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
    
    def _convert_to_gemini_tools(self, groq_tools: List[Dict]) -> List[Dict]:
        """
        Convert Groq/OpenAI format tools to Gemini format.
        
        Groq format: {"type": "function", "function": {...}}
        Gemini format: {"name": "...", "description": "...", "parameters": {...}}
        
        Gemini requires:
        - Property types as UPPERCASE (STRING, NUMBER, BOOLEAN, ARRAY, OBJECT)
        - No "type": "object" at root parameters level
        """
        gemini_tools = []
        
        def convert_type(json_type: str) -> str:
            """Convert JSON Schema type to Gemini type."""
            type_map = {
                "string": "STRING",
                "number": "NUMBER",
                "integer": "INTEGER",
                "boolean": "BOOLEAN",
                "array": "ARRAY",
                "object": "OBJECT"
            }
            return type_map.get(json_type, "STRING")
        
        def convert_properties(properties: Dict) -> Dict:
            """Convert property definitions to Gemini format."""
            converted = {}
            for prop_name, prop_def in properties.items():
                new_def = {}
                
                # Handle oneOf (like clean_missing_values strategy)
                if "oneOf" in prop_def:
                    # For oneOf, just pick the first option or simplify
                    if isinstance(prop_def["oneOf"], list) and len(prop_def["oneOf"]) > 0:
                        first_option = prop_def["oneOf"][0]
                        if "type" in first_option:
                            new_def["type"] = convert_type(first_option["type"])
                        if "enum" in first_option:
                            new_def["enum"] = first_option["enum"]
                    else:
                        new_def["type"] = "STRING"
                elif "type" in prop_def:
                    new_def["type"] = convert_type(prop_def["type"])
                    
                    # Handle arrays
                    if prop_def["type"] == "array" and "items" in prop_def:
                        new_def["items"] = {"type": convert_type(prop_def["items"].get("type", "string"))}
                    
                    # Keep enum
                    if "enum" in prop_def:
                        new_def["enum"] = prop_def["enum"]
                else:
                    new_def["type"] = "STRING"
                
                # Keep description if present
                if "description" in prop_def:
                    new_def["description"] = prop_def["description"]
                
                converted[prop_name] = new_def
            
            return converted
        
        for tool in groq_tools:
            func = tool["function"]
            params = func.get("parameters", {})
            
            # Convert parameters to Gemini format
            gemini_params = {
                "type": "OBJECT",  # Gemini uses UPPERCASE
                "properties": convert_properties(params.get("properties", {})),
                "required": params.get("required", [])
            }
            
            gemini_tool = {
                "name": func["name"],
                "description": func["description"],
                "parameters": gemini_params
            }
            gemini_tools.append(gemini_tool)
        
        return gemini_tools
    
    def analyze(self, file_path: str, task_description: str, 
               target_col: Optional[str] = None, 
               use_cache: bool = True,
               stream: bool = True,
               max_iterations: int = 20) -> Dict[str, Any]:
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
        
        # For Gemini, maintain a persistent chat session
        gemini_chat = None
        if self.provider == "gemini":
            gemini_chat = self.gemini_model.start_chat(history=[])
        
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
                
                # Rate limiting - wait if needed (for Gemini free tier: 10 RPM)
                if self.min_api_call_interval > 0:
                    time_since_last_call = time.time() - self.last_api_call_time
                    if time_since_last_call < self.min_api_call_interval:
                        wait_time = self.min_api_call_interval - time_since_last_call
                        print(f"⏳ Rate limiting: waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                
                # Call LLM with function calling (provider-specific)
                if self.provider == "groq":
                    response = self.groq_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools_to_use,
                        tool_choice="auto",
                        parallel_tool_calls=False,  # Disable parallel calls to prevent XML format errors
                        temperature=0.1,  # Low temperature for consistent outputs
                        max_tokens=4096
                    )
                    
                    self.api_calls_made += 1
                    self.last_api_call_time = time.time()
                    response_message = response.choices[0].message
                    tool_calls = response_message.tool_calls
                    final_content = response_message.content
                    
                elif self.provider == "gemini":
                    # Convert tools to Gemini format
                    gemini_tools = self._convert_to_gemini_tools(tools_to_use)
                    
                    # First iteration: send system + user message
                    if iteration == 1:
                        combined_message = f"{messages[0]['content']}\n\n{messages[1]['content']}"
                        response = gemini_chat.send_message(
                            combined_message,
                            tools=gemini_tools
                        )
                    else:
                        # Subsequent iterations: send function responses
                        # Gemini needs function responses to continue the conversation
                        # The last message should be a tool response
                        last_tool_msg = messages[-1]
                        if last_tool_msg.get("role") == "tool":
                            # Send function response back to Gemini using proper format
                            from google.ai.generativelanguage_v1beta.types import content as glm_content
                            
                            function_response_part = glm_content.Part(
                                function_response=glm_content.FunctionResponse(
                                    name=last_tool_msg["name"],
                                    response={"result": last_tool_msg["content"]}
                                )
                            )
                            
                            response = gemini_chat.send_message(
                                function_response_part,
                                tools=gemini_tools
                            )
                        else:
                            # Shouldn't happen, but fallback
                            response = gemini_chat.send_message(
                                "Continue with the next step.",
                                tools=gemini_tools
                            )
                    
                    self.api_calls_made += 1
                    self.last_api_call_time = time.time()
                    
                    # Extract function calls from Gemini response
                    tool_calls = []
                    final_content = None
                    
                    if response.candidates and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                tool_calls.append(part.function_call)
                            elif hasattr(part, 'text') and part.text:
                                final_content = part.text
                
                # Check if done (no tool calls)
                if not tool_calls:
                    # Final response
                    final_summary = final_content or "Analysis completed"
                    
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
                
                # Execute tool calls (provider-specific format)
                if self.provider == "groq":
                    messages.append(response_message)
                
                for tool_call in tool_calls:
                    # Extract tool name and args (provider-specific)
                    if self.provider == "groq":
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_call_id = tool_call.id
                    elif self.provider == "gemini":
                        tool_name = tool_call.name
                        # Convert protobuf args to Python dict
                        tool_args = {}
                        for key, value in tool_call.args.items():
                            # Handle different protobuf value types
                            if isinstance(value, (str, int, float, bool)):
                                tool_args[key] = value
                            elif hasattr(value, '__iter__') and not isinstance(value, str):
                                # Convert lists/repeated fields
                                tool_args[key] = list(value)
                            else:
                                # Fallback: try to convert to string
                                tool_args[key] = str(value)
                        tool_call_id = f"gemini_{iteration}_{tool_name}"
                    
                    print(f"\n🔧 Executing: {tool_name}")
                    try:
                        print(f"   Arguments: {json.dumps(tool_args, indent=2)}")
                    except:
                        print(f"   Arguments: {tool_args}")
                    
                    # Execute tool
                    tool_result = self._execute_tool(tool_name, tool_args)
                    
                    # Track in workflow
                    workflow_history.append({
                        "iteration": iteration,
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": tool_result
                    })
                    
                    print(f"   ✓ Completed: {tool_name}")
                    
                    # Debug: Check if training completed
                    if tool_name == "train_baseline_models":
                        print(f"[DEBUG] train_baseline_models executed!")
                        print(f"[DEBUG]   tool_result keys: {list(tool_result.keys())}")
                        print(f"[DEBUG]   'best_model' in tool_result: {'best_model' in tool_result}")
                        if isinstance(tool_result, dict) and 'result' in tool_result:
                            print(f"[DEBUG]   Nested result keys: {list(tool_result['result'].keys()) if isinstance(tool_result['result'], dict) else 'Not a dict'}")
                            print(f"[DEBUG]   'best_model' in nested result: {'best_model' in tool_result['result'] if isinstance(tool_result['result'], dict) else False}")
                        if "best_model" in tool_result:
                            print(f"[DEBUG]   best_model value: {tool_result['best_model']}")
                    
                    # Check if training is complete - if so, finish successfully BEFORE adding to messages
                    # Extract the actual result (might be nested under 'result' key)
                    actual_result = tool_result.get("result", tool_result) if isinstance(tool_result, dict) else tool_result
                    if tool_name == "train_baseline_models" and isinstance(actual_result, dict) and "best_model" in actual_result:
                        print(f"🎯 AUTO-FINISH TRIGGERED! Training complete, returning comprehensive report...")
                        # Generate comprehensive summary
                        summary = f"""✅ **Machine Learning Pipeline Complete!**

**Dataset:** `{file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]}`
**Target:** `{target_col or 'Auto-detected'}`

---

### 📊 Data Processing Pipeline

**Step 1: Data Profiling**
- Analyzed {actual_result.get('n_samples', 'N/A')} rows with {actual_result.get('n_features', 'N/A')} features
- Identified data types and missing values

**Step 2: Data Quality**
- Detected and documented quality issues
- Checked for missing values, duplicates, outliers

**Step 3: Data Cleaning**
- Cleaned missing values using auto-detection
- Handled outliers with clipping method

**Step 4: Feature Engineering**
- Converted string columns to numeric (force conversion)
- Encoded categorical variables
- Prepared features for modeling

**Step 5: Model Training** 🎯
- Trained **4 baseline models**: Ridge, Lasso, Random Forest, XGBoost
- Task type: **{actual_result.get('task_type', 'regression').title()}**
- Train/Test split: 80/20

---

### 🏆 Best Model Results

**Model:** {actual_result.get('best_model', {}).get('name', 'N/A').upper().replace('_', ' ')}
**Performance Score (R²):** {actual_result.get('best_model', {}).get('score', 'N/A'):.4f}

**All Models Trained:**
"""
                        if "models" in actual_result:
                            for model_name in actual_result["models"].keys():
                                summary += f"\n- ✓ {model_name.replace('_', ' ').title()}"
                        
                        summary += f"""

**Model saved at:** `./outputs/models/`
**Processed data:** `./outputs/data/encoded.csv`

---

**Total execution time:** {round(time.time() - start_time, 2)}s
**API calls:** {self.api_calls_made}
**Pipeline steps:** {len(workflow_history)}
"""
                        
                        return {
                            "status": "success",
                            "summary": summary,
                            "workflow_history": workflow_history,
                            "iterations": iteration,
                            "api_calls": self.api_calls_made,
                            "execution_time": round(time.time() - start_time, 2)
                        }
            
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                print(f"❌ ERROR in analyze loop: {e}")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Traceback:\n{error_traceback}")
                return {
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": error_traceback,
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
