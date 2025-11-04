"""
AI Agent Data Scientist - Interactive Chat UI
==============================================

A simple web interface to interact with your AI Agent.
Upload datasets, ask questions, and get AI-powered insights!
"""

import gradio as gr
import sys
import os
from pathlib import Path
import traceback

# Add src to path
sys.path.append('src')

from tools.data_profiling import profile_dataset, detect_data_quality_issues
from tools.model_training import train_baseline_models

# Try to import AI agent (optional)
try:
    from orchestrator import DataScienceCopilot
    agent = DataScienceCopilot()
    AI_ENABLED = True
    print("✅ AI Agent loaded successfully!")
    print(f"📊 Model: {agent.model}")
    print(f"🔧 Tools available: {len(agent.tool_functions)}")
except Exception as e:
    print(f"ℹ️  Running in manual mode (AI agent not available)")
    print(f"   Error: {str(e)}")
    print("💡 You can still use all the quick actions and tools!")
    AI_ENABLED = False
    agent = None

# Store uploaded file path
current_file = None
current_profile = None


def analyze_dataset(file, user_message, history):
    """Process uploaded dataset and user message."""
    global current_file, current_profile
    
    # Initialize history if None
    if history is None:
        history = []
    
    # Debug: Log the call
    print(f"[DEBUG] analyze_dataset called - file: {file is not None}, message: '{user_message}', current_file: {current_file}")
    
    try:
        # If a new file is uploaded (and it's different from current)
        if file is not None and (current_file is None or file.name != current_file):
            print(f"[DEBUG] Processing new file upload: {file.name}")
            current_file = file.name
            
            # Profile the dataset
            response = f"📊 **Dataset Uploaded Successfully!**\n\n"
            response += f"**File:** {Path(current_file).name}\n\n"
            
            # Get basic profile
            profile = profile_dataset(current_file)
            current_profile = profile
            
            response += f"**Dataset Overview:**\n"
            response += f"- Rows: {profile['shape']['rows']:,}\n"
            response += f"- Columns: {profile['shape']['columns']}\n"
            
            # Handle memory_usage (can be float or dict)
            memory = profile.get('memory_usage', 0)
            if isinstance(memory, dict):
                memory = memory.get('total_mb', 0)
            response += f"- Memory: {memory:.2f} MB\n\n"
            
            response += f"**Column Types:**\n"
            response += f"- Numeric: {len(profile['column_types']['numeric'])} columns\n"
            response += f"- Categorical: {len(profile['column_types']['categorical'])} columns\n"
            response += f"- Datetime: {len(profile['column_types']['datetime'])} columns\n\n"
            
            # Check data quality
            quality = detect_data_quality_issues(current_file)
            if quality['critical']:
                response += f"🔴 **Critical Issues:** {len(quality['critical'])}\n"
                for issue in quality['critical'][:3]:
                    response += f"  - {issue['message']}\n"
            if quality['warning']:
                response += f"🟡 **Warnings:** {len(quality['warning'])}\n"
                for issue in quality['warning'][:3]:
                    response += f"  - {issue['message']}\n"
            
            response += f"\n\n💬 **What would you like to do with this dataset?**\n\n"
            response += "You can ask me to:\n"
            response += "- Train a classification or regression model\n"
            response += "- Analyze specific columns\n"
            response += "- Detect outliers\n"
            response += "- Engineer features\n"
            response += "- Generate predictions\n"
            response += "- And much more!\n"
            
            # Add or replace message in history
            if history and len(history) > 0 and history[-1][0] is None:
                history[-1] = (None, response)
            else:
                history.append((None, response))
            yield history, ""
            return
        
        # If user sends a message about the current file
        print(f"[DEBUG] Checking message conditions: user_message={bool(user_message and user_message.strip())}, current_file={bool(current_file)}")
        if user_message and user_message.strip() and current_file:
            print(f"[DEBUG] User message detected. AI_ENABLED={AI_ENABLED}, agent={agent is not None}")
            if AI_ENABLED and agent:
                print(f"[DEBUG] Entering AI Agent block...")
                try:
                    # Show immediate processing message
                    print(f"🤖 AI Agent analyzing: {user_message}")
                    history.append((user_message, "🤖 **AI Agent is thinking...**\n\n⏳ Analyzing your request and planning the workflow..."))
                    yield history, ""
                    
                    # Use the AI agent to process the request
                    print(f"📂 File path: {current_file}")
                    print(f"📝 Task: {user_message}")
                    print(f"🚀 Calling agent.analyze()...")
                    
                    agent_response = agent.analyze(
                        file_path=current_file,
                        task_description=user_message,
                        use_cache=False,  # Disable cache to avoid dict hashing issues
                        stream=False
                    )
                    
                    print(f"✅ Agent response received: {agent_response.get('status', 'unknown')}")
                    
                    # Format the response
                    if agent_response.get('status') == 'success':
                        response = f"🤖 **AI Agent Analysis Complete!**\n\n"
                        response += f"{agent_response.get('summary', '')}\n\n"
                        
                        if 'workflow_history' in agent_response and agent_response['workflow_history']:
                            response += f"**Tools Executed:** {len(agent_response['workflow_history'])}\n"
                            response += f"**Iterations:** {agent_response.get('iterations', 0)}\n"
                            response += f"**Time:** {agent_response.get('execution_time', 0)}s\n\n"
                            
                            # Show tool execution summary
                            response += "**Workflow:**\n"
                            for step in agent_response['workflow_history'][-5:]:  # Show last 5 steps
                                tool_name = step['tool']
                                success = step['result'].get('success', False)
                                icon = "✅" if success else "❌"
                                response += f"{icon} {tool_name}\n"
                    else:
                        response = f"⚠️ **AI Agent Status:** {agent_response.get('status', 'unknown')}\n\n"
                        response += f"{agent_response.get('message', agent_response.get('error', 'Unknown error'))}\n"
                    
                    # Replace loading message safely
                    if history and len(history) > 0 and history[-1][0] == user_message:
                        history[-1] = (user_message, response)
                    else:
                        history.append((user_message, response))
                    yield history, ""
                    return
                except Exception as e:
                    import sys
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    response = f"⚠️ **AI Agent Error:**\n\n"
                    response += f"**Error Type:** {exc_type.__name__}\n\n"
                    response += f"**Error Message:** {str(e)}\n\n"
                    response += f"**Full Traceback:**\n```python\n{traceback.format_exc()}\n```\n\n"
                    response += "💡 **Fallback Options:**\n"
                    response += "- Use the **Quick Train** feature on the right\n"
                    response += "- Try manual commands: `profile`, `quality`, `columns`\n"
                    # Replace loading message safely
                    if history and len(history) > 0 and history[-1][0] == user_message:
                        history[-1] = (user_message, response)
                    else:
                        history.append((user_message, response))
                    yield history, ""
                    return
            else:
                # Manual mode - Handle commands directly
                user_msg_lower = user_message.lower().strip()
                
                # Handle simple commands manually
                if 'profile' in user_msg_lower:
                    response = "📊 **Dataset Profile:**\n\n"
                    if current_profile:
                        response += f"**Shape:** {current_profile['shape']['rows']:,} rows × {current_profile['shape']['columns']} columns\n\n"
                        response += f"**Column Types:**\n"
                        response += f"- Numeric: {len(current_profile['column_types']['numeric'])} columns\n"
                        response += f"- Categorical: {len(current_profile['column_types']['categorical'])} columns\n"
                        response += f"- Datetime: {len(current_profile['column_types']['datetime'])} columns\n\n"
                        response += f"**Overall Stats:**\n"
                        response += f"- Total cells: {current_profile['overall_stats']['total_cells']:,}\n"
                        response += f"- Null values: {current_profile['overall_stats']['total_nulls']} ({current_profile['overall_stats']['null_percentage']:.1f}%)\n"
                        response += f"- Duplicates: {current_profile['overall_stats']['duplicate_rows']}\n"
                    else:
                        response += "Profile information is available at the top of the chat!"
                        
                elif 'quality' in user_msg_lower or 'issues' in user_msg_lower:
                    quality = detect_data_quality_issues(current_file)
                    response = "🔍 **Data Quality Report:**\n\n"
                    
                    if quality['critical']:
                        response += f"🔴 **Critical Issues:** {len(quality['critical'])}\n"
                        for issue in quality['critical']:
                            response += f"  • {issue['message']}\n"
                        response += "\n"
                    
                    if quality['warning']:
                        response += f"🟡 **Warnings:** {len(quality['warning'])}\n"
                        for issue in quality['warning'][:5]:  # Show first 5
                            response += f"  • {issue['message']}\n"
                        if len(quality['warning']) > 5:
                            response += f"  • ... and {len(quality['warning']) - 5} more\n"
                        response += "\n"
                    
                    if quality['info']:
                        response += f"🔵 **Info:** {len(quality['info'])} observations\n"
                    
                    if not quality['critical'] and not quality['warning'] and not quality['info']:
                        response += "✅ No issues detected! Your data looks good.\n"
                        
                elif 'columns' in user_msg_lower or 'column' in user_msg_lower:
                    if current_profile:
                        response = "📋 **Dataset Columns:**\n\n"
                        for col, info in current_profile['columns'].items():
                            nulls = info.get('null_count', 0)
                            null_pct = (nulls / current_profile['shape']['rows'] * 100) if current_profile['shape']['rows'] > 0 else 0
                            response += f"• **{col}** ({info['type']})\n"
                            response += f"  - Nulls: {nulls} ({null_pct:.1f}%)\n"
                            if 'unique' in info:
                                response += f"  - Unique: {info['unique']}\n"
                    else:
                        response = "📋 **Columns:** Please upload a file first to see column information."
                
                elif 'help' in user_msg_lower:
                    response = "💡 **Available Commands:**\n\n"
                    response += "**Manual Commands:**\n"
                    response += "• `profile` - Show detailed dataset statistics\n"
                    response += "• `quality` - Check data quality issues\n"
                    response += "• `columns` - List all columns with details\n"
                    response += "• `help` - Show this help message\n\n"
                    response += "**Quick Actions:**\n"
                    response += "• Use the **Quick Train** panel on the right to train models\n"
                    response += "• Check **Dataset Info** in the sidebar for quick stats\n"
                
                else:
                    # Default response for unrecognized commands
                    response = f"💬 **You said:** {user_message}\n\n"
                    response += "⚠️ AI agent is not available. I can respond to these commands:\n\n"
                    response += "• `profile` - Show detailed statistics\n"
                    response += "• `quality` - Check data quality\n"
                    response += "• `columns` - List all columns\n"
                    response += "• `help` - Show available commands\n\n"
                    response += "**Or use Quick Train** on the right to train models directly!\n"
                
                # Replace loading message safely
                if history and len(history) > 0 and history[-1][0] == user_message:
                    history[-1] = (user_message, response)
                else:
                    history.append((user_message, response))
                yield history, ""
                return
        
        # If no file is uploaded yet
        if user_message and user_message.strip() and not current_file:
            response = "⚠️ **Please upload a dataset first!**\n\n"
            response += "Click the 'Upload Dataset' button above and select a CSV or Parquet file."
            # Replace loading message safely
            if history and len(history) > 0 and history[-1][0] == user_message:
                history[-1] = (user_message, response)
            else:
                history.append((user_message, response))
            yield history, ""
            return
            
    except Exception as e:
        error_msg = f"❌ **Error:** {str(e)}\n\n"
        error_msg += "**Traceback:**\n```\n" + traceback.format_exc() + "\n```"
        if user_message:
            if history and history[-1][0] == user_message:
                history[-1] = (user_message, error_msg)
            else:
                history.append((user_message, error_msg))
        else:
            history.append((None, error_msg))
        yield history, ""
        return
    
    # Default return if nothing matched
    yield history, ""


def quick_profile(file):
    """Quick profile display in the sidebar."""
    if file is None:
        return "No file uploaded yet."
    
    try:
        profile = profile_dataset(file.name)
        
        info = f"**{Path(file.name).name}**\n\n"
        info += f"📊 {profile['shape']['rows']:,} rows × {profile['shape']['columns']} cols\n\n"
        info += f"**Columns:**\n"
        for col, col_info in list(profile['columns'].items())[:10]:
            info += f"- {col} ({col_info['type']})\n"
        
        if len(profile['columns']) > 10:
            info += f"- ... and {len(profile['columns']) - 10} more\n"
        
        return info
    except Exception as e:
        return f"Error: {str(e)}"


def train_model_ui(file, target_col, model_type, test_size, progress=gr.Progress()):
    """Train a model directly from the UI."""
    if file is None:
        return "⚠️ Please upload a dataset first!"
    
    if not target_col:
        return "⚠️ Please specify a target column!"
    
    # Clean up the target column name - remove surrounding quotes if present
    target_col = target_col.strip().strip("'").strip('"')
    
    try:
        # Show progress
        progress(0, desc="🔄 Loading dataset...")
        yield "⏳ **Training in progress...**\n\n📊 Loading dataset..."
        
        import time
        time.sleep(0.5)  # Brief pause for UI feedback
        
        progress(0.2, desc="🔄 Preparing data...")
        yield "⏳ **Training in progress...**\n\n📊 Dataset loaded\n🔄 Preparing data..."
        
        time.sleep(0.3)
        # Determine problem type
        problem_type = "classification" if model_type == "Classification" else "regression"
        
        progress(0.4, desc="🤖 Training models...")
        yield "⏳ **Training in progress...**\n\n📊 Dataset loaded\n✅ Data prepared\n🤖 Training multiple models..."
        
        # Train baseline models
        result = train_baseline_models(
            file.name,
            target_col=target_col,
            task_type=problem_type,
            test_size=test_size
        )
        
        progress(0.9, desc="📊 Evaluating results...")
        
        # Check if training was successful
        if result.get('status') == 'error':
            yield f"❌ **Training Failed**\n\n{result.get('message', 'Unknown error')}"
            return
        
        if 'best_model' not in result:
            yield f"❌ **Training Failed**\n\nNo models were successfully trained. Result: {result}"
            return
        
        # Get the best model
        best_model_name = result['best_model']['name']
        if not best_model_name:
            yield f"❌ **Training Failed**\n\nNo model could be selected as best model."
            return
            
        best_model_info = result['models'][best_model_name]
        best_metrics = best_model_info.get('test_metrics', {})
        
        output = f"✅ **Model Trained Successfully!**\n\n"
        output += f"**Best Model:** {best_model_name}\n\n"
        
        if problem_type == "classification":
            output += f"**Test Metrics:**\n"
            output += f"- Accuracy: {best_metrics.get('accuracy', 0):.3f}\n"
            output += f"- Precision: {best_metrics.get('precision', 0):.3f}\n"
            output += f"- Recall: {best_metrics.get('recall', 0):.3f}\n"
            output += f"- F1 Score: {best_metrics.get('f1', 0):.3f}\n\n"
        else:
            output += f"**Test Metrics:**\n"
            output += f"- R² Score: {best_metrics.get('r2', 0):.3f}\n"
            output += f"- RMSE: {best_metrics.get('rmse', 0):.3f}\n"
            output += f"- MAE: {best_metrics.get('mae', 0):.3f}\n\n"
        
        output += f"**All Models Tested:**\n"
        for model_name, model_info in result['models'].items():
            if 'test_metrics' in model_info:
                test_metrics = model_info['test_metrics']
                if problem_type == "classification":
                    f1 = test_metrics.get('f1', 0)
                    output += f"- {model_name}: {f1:.3f} F1 score\n"
                else:
                    r2 = test_metrics.get('r2', 0)
                    output += f"- {model_name}: {r2:.3f} R²\n"
            elif 'status' in model_info and model_info['status'] == 'error':
                output += f"- {model_name}: ❌ {model_info.get('message', 'Error')}\n"
        
        progress(1.0, desc="✅ Complete!")
        yield output
            
    except Exception as e:
        yield f"❌ **Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def clear_conversation():
    """Clear the conversation and reset state."""
    global current_file, current_profile
    current_file = None
    current_profile = None
    return [], None, ""


# Custom CSS for better visual feedback
custom_css = """
.status-box {
    padding: 10px;
    border-radius: 5px;
    background: linear-gradient(90deg, #e8f5e9 0%, #c8e6c9 100%);
    margin-bottom: 10px;
    text-align: center;
    font-weight: bold;
}
"""

# Create the Gradio interface
with gr.Blocks(title="AI Agent Data Scientist", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""
    # 🤖 AI Agent Data Scientist
    
    Upload your dataset and chat with the AI agent to perform data science tasks!
    
    **Features:**
    - 📊 Automatic dataset profiling
    - 🤖 Natural language queries
    - 🎯 Model training (classification & regression)
    - 🔍 Data quality analysis
    - 📈 Feature engineering
    - And 40+ more tools!
    """)
    
    with gr.Row():
        # Left column - Main chat interface
        with gr.Column(scale=2):
            # Status indicator
            status_box = gr.Markdown("🟢 **Ready** - Upload a dataset to begin", elem_classes=["status-box"])
            
            chatbot = gr.Chatbot(
                label="Chat with AI Agent",
                height=450,
                show_label=True,
                avatar_images=(None, "🤖")
            )
            
            with gr.Row():
                file_upload = gr.File(
                    label="📁 Upload Dataset (CSV/Parquet)",
                    file_types=[".csv", ".parquet"],
                    type="filepath"
                )
            
            with gr.Row():
                user_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type: profile, quality, columns, or help",
                    lines=2,
                    scale=4
                )
                submit_btn = gr.Button("📤 Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        # Right column - Quick actions and info
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Dataset Info")
            dataset_info = gr.Markdown("Upload a dataset to see information here.")
            
            gr.Markdown("## 🎯 Quick Train")
            with gr.Group():
                target_column = gr.Textbox(
                    label="Target Column",
                    placeholder="e.g., 'price', 'class', 'label'"
                )
                model_type_choice = gr.Radio(
                    ["Classification", "Regression"],
                    label="Model Type",
                    value="Classification"
                )
                test_size_slider = gr.Slider(
                    0.1, 0.5, 0.3,
                    label="Test Size",
                    step=0.05
                )
                train_btn = gr.Button("🚀 Train Model", variant="primary")
            
            training_output = gr.Markdown("Training results will appear here.")
            
            gr.Markdown("""
            ## 💡 Example Queries
            
            - "Train a classification model to predict [target]"
            - "Show me statistics for [column]"
            - "Detect outliers in the dataset"
            - "What are the most important features?"
            - "Generate a quality report"
            - "Create polynomial features"
            - "Balance the dataset using SMOTE"
            """)
    
    # Event handlers with streaming support
    submit_btn.click(
        fn=analyze_dataset,
        inputs=[file_upload, user_input, chatbot],
        outputs=[chatbot, user_input],
        show_progress="full"  # Show progress bar
    )
    
    user_input.submit(
        fn=analyze_dataset,
        inputs=[file_upload, user_input, chatbot],
        outputs=[chatbot, user_input],
        show_progress="full"
    )
    
    file_upload.change(
        fn=analyze_dataset,
        inputs=[file_upload, gr.Textbox(value="", visible=False), chatbot],
        outputs=[chatbot, user_input],
        show_progress="full"
    ).then(
        fn=quick_profile,
        inputs=[file_upload],
        outputs=[dataset_info]
    )
    
    train_btn.click(
        fn=train_model_ui,
        inputs=[file_upload, target_column, model_type_choice, test_size_slider],
        outputs=[training_output],
        show_progress="full"  # Show progress bar
    )
    
    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, file_upload, user_input]
    )

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Starting AI Agent Data Scientist Chat UI...")
    print("=" * 70)
    print("\n🌐 The UI will open in your browser automatically.")
    print("💡 If it doesn't, copy the URL shown below.\n")
    
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",
        server_port=7866,  # Changed port to avoid conflict
        show_error=True
    )
