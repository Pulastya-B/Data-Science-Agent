"""
FastAPI Application for Google Cloud Run
Thin HTTP wrapper around DataScienceCopilot - No logic changes, just API exposure.
Serves React frontend and provides streaming chat API.
"""

import os
import sys
import tempfile
import shutil
import json
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from orchestrator import DataScienceCopilot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Data Science Agent API",
    description="Cloud Run wrapper for autonomous data science workflows",
    version="2.0.0"
)

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent once (singleton pattern for stateless service)
# Agent itself is stateless - no conversation memory between requests
agent: Optional[DataScienceCopilot] = None


@app.on_event("startup")
async def startup_event():
    """Initialize DataScienceCopilot on service startup."""
    global agent
    try:
        logger.info("Initializing DataScienceCopilot...")
        agent = DataScienceCopilot(
            reasoning_effort="medium",
            provider=os.getenv("LLM_PROVIDER", "groq")
        )
        logger.info(f"✅ Agent initialized with provider: {agent.provider}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Data Science Agent API",
        "status": "healthy",
        "provider": agent.provider if agent else "not initialized",
        "tools_available": len(agent.tool_functions) if agent else 0
    }


@app.get("/health")
async def health_check():
    """
    Health check for Cloud Run.
    Returns 200 if service is ready to accept requests.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "status": "healthy",
        "agent_ready": True,
        "provider": agent.provider,
        "tools_count": len(agent.tool_functions)
    }


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint (JSON body)."""
    task_description: str
    target_col: Optional[str] = None
    use_cache: bool = True
    max_iterations: int = 20


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    messages: List[ChatMessage]
    file_path: Optional[str] = None
    stream: bool = True


# ==================== CHAT API ====================

@app.post("/api/chat")
async def chat_completion(request: ChatRequest):
    """
    Chat completion endpoint - connects frontend to Data Science Agent.
    Supports streaming responses for real-time updates.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Get the last user message
    user_messages = [m for m in request.messages if m.role == 'user']
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")
    
    last_message = user_messages[-1].content
    logger.info(f"Chat request: {last_message[:100]}...")
    
    async def generate_response():
        """Stream response chunks."""
        try:
            # Check if this is a data science task with file
            if request.file_path and os.path.exists(request.file_path):
                # Run the agent with the file
                result = agent.analyze(
                    file_path=request.file_path,
                    task_description=last_message,
                    max_iterations=10
                )
                
                # Format response with HTML reports embedded
                if result.get("status") == "success":
                    response_text = f"## Analysis Complete\n\n"
                    response_text += f"**Summary:** {result.get('summary', 'Analysis completed successfully.')}\n\n"
                    
                    if result.get("tools_used"):
                        response_text += f"**Tools Used:** {', '.join(result['tools_used'])}\n\n"
                    
                    if result.get("insights"):
                        response_text += f"### Insights\n{result['insights']}\n\n"
                    
                    # Stream the text response first
                    for i in range(0, len(response_text), 50):
                        chunk = response_text[i:i+50]
                        yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
                        await asyncio.sleep(0.02)
                    
                    # Check for HTML reports and send them as complete chunks
                    html_reports = []
                    
                    # Get absolute paths to check multiple possible output directories
                    base_dir = Path(__file__).parent.parent.parent  # Project root
                    api_dir = Path(__file__).parent  # src/api directory
                    
                    # Check both possible report locations
                    possible_dirs = [
                        base_dir / "outputs" / "reports",  # Project root outputs
                        api_dir / "outputs" / "reports",   # src/api outputs (where agent actually saves)
                    ]
                    
                    logger.info(f"Checking for reports in multiple locations:")
                    for reports_dir in possible_dirs:
                        logger.info(f"  - {reports_dir} (exists: {reports_dir.exists()})")
                    
                    # Find all HTML files from all locations
                    all_html_files = []
                    for reports_dir in possible_dirs:
                        if reports_dir.exists():
                            html_files = list(reports_dir.glob('*.html'))
                            all_html_files.extend(html_files)
                            logger.info(f"  Found {len(html_files)} HTML files in {reports_dir}")
                    
                    if all_html_files:
                        # Sort by modification time (most recent first)
                        all_html_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        logger.info(f"Total {len(all_html_files)} HTML report(s) found: {[f.name for f in all_html_files]}")
                        
                        # Load the most recent reports (limit to 2 to avoid huge payloads)
                        for html_file in all_html_files[:2]:
                            try:
                                # Check if file was created/modified in last 5 minutes (likely from this run)
                                file_age = time.time() - html_file.stat().st_mtime
                                if file_age < 300:  # 5 minutes
                                    with open(html_file, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                        report_name = html_file.stem.replace('_', ' ').title()
                                        html_reports.append((report_name, html_content))
                                        logger.info(f"✅ Loaded {html_file.name} ({len(html_content)} bytes, {file_age:.1f}s old)")
                                else:
                                    logger.info(f"⏭️  Skipping {html_file.name} (too old: {file_age:.1f}s)")
                            except Exception as e:
                                logger.error(f"Error reading {html_file}: {e}")
                    
                    logger.info(f"Total HTML reports to send: {len(html_reports)}")
                    
                    # Send HTML reports as complete chunks with special marker
                    if html_reports:
                        yield f"data: {json.dumps({'content': '\\n---\\n\\n', 'done': False})}\n\n"
                        
                        for report_name, html_content in html_reports:
                            # Send report title
                            yield f"data: {json.dumps({'content': f'### {report_name}\\n\\n', 'done': False})}\n\n"
                            
                            # Send entire HTML report as ONE chunk with special html_report field
                            logger.info(f"Sending {report_name} ({len(html_content)} bytes)")
                            yield f"data: {json.dumps({'html_report': html_content, 'done': False})}\n\n"
                            await asyncio.sleep(0.1)
                    else:
                        logger.warning("No HTML reports found to send!")
                else:
                    response_text = f"Analysis encountered an issue: {result.get('error', 'Unknown error')}"
                    
                    # Stream error message
                    for i in range(0, len(response_text), 50):
                        chunk = response_text[i:i+50]
                        yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
                        await asyncio.sleep(0.02)
                
                yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
            else:
                # Regular chat - use the agent's chat method
                response = agent.chat(last_message)
                
                # Stream in chunks for smooth UI
                for i in range(0, len(response), 30):
                    chunk = response[i:i+30]
                    yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
                    await asyncio.sleep(0.01)
                
                yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
                
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    
    if request.stream:
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    else:
        # Non-streaming response
        try:
            if request.file_path and os.path.exists(request.file_path):
                result = agent.analyze(
                    file_path=request.file_path,
                    task_description=last_message,
                    max_iterations=10
                )
                return JSONResponse(content={"success": True, "result": result})
            else:
                response = agent.chat(last_message)
                return JSONResponse(content={"success": True, "content": response})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file for analysis.
    Returns a temporary file path that can be used in subsequent chat messages.
    """
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.parquet') or filename.endswith('.xlsx')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Supported: CSV, Parquet, Excel"
        )
    
    temp_dir = Path("/tmp") / "data_science_agent" / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_file_path = temp_dir / file.filename
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"File uploaded: {file.filename} ({file_size} bytes)")
        
        return JSONResponse(content={
            "success": True,
            "file_path": str(temp_file_path),
            "filename": file.filename,
            "size": file_size
        })
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== EXISTING ENDPOINTS ====================
    use_cache: bool = True
    max_iterations: int = 20


@app.post("/run")
async def run_analysis(
    file: UploadFile = File(..., description="Dataset file (CSV or Parquet)"),
    task_description: str = Form(..., description="Natural language task description"),
    target_col: Optional[str] = Form(None, description="Target column name for prediction"),
    use_cache: bool = Form(True, description="Enable caching for expensive operations"),
    max_iterations: int = Form(20, description="Maximum workflow iterations")
) -> JSONResponse:
    """
    Run complete data science workflow on uploaded dataset.
    
    This is a thin wrapper - all logic lives in DataScienceCopilot.analyze().
    
    Args:
        file: CSV or Parquet file upload
        task_description: Natural language description of the task
        target_col: Optional target column for ML tasks
        use_cache: Whether to use cached results
        max_iterations: Maximum number of workflow steps
        
    Returns:
        JSON response with analysis results, workflow history, and execution stats
        
    Example:
        ```bash
        curl -X POST http://localhost:8080/run \
          -F "file=@data.csv" \
          -F "task_description=Analyze this dataset and predict house prices" \
          -F "target_col=price"
        ```
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Validate file format
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.parquet')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only CSV and Parquet files are supported."
        )
    
    # Use /tmp for Cloud Run (ephemeral storage)
    temp_dir = Path("/tmp") / "data_science_agent"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_file_path = None
    
    try:
        # Save uploaded file to temporary location
        temp_file_path = temp_dir / file.filename
        logger.info(f"Saving uploaded file to: {temp_file_path}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved successfully: {file.filename} ({os.path.getsize(temp_file_path)} bytes)")
        
        # Call existing agent logic - NO CHANGES to orchestrator
        logger.info(f"Starting analysis with task: {task_description}")
        result = agent.analyze(
            file_path=str(temp_file_path),
            task_description=task_description,
            target_col=target_col,
            use_cache=use_cache,
            max_iterations=max_iterations
        )
        
        logger.info(f"Analysis completed: {result.get('status')}")
        
        # Return result as-is from orchestrator
        return JSONResponse(
            content={
                "success": result.get("status") == "success",
                "result": result,
                "metadata": {
                    "filename": file.filename,
                    "task": task_description,
                    "target": target_col,
                    "provider": agent.provider
                }
            },
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "error_type": type(e).__name__,
                "message": "Analysis workflow failed. Check logs for details."
            }
        )
    
    finally:
        # Cleanup temporary file
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


@app.post("/profile")
async def profile_dataset(
    file: UploadFile = File(..., description="Dataset file (CSV or Parquet)")
) -> JSONResponse:
    """
    Quick dataset profiling without full workflow.
    
    Returns basic statistics, data types, and quality issues.
    Useful for initial data exploration without running full analysis.
    
    Example:
        ```bash
        curl -X POST http://localhost:8080/profile \
          -F "file=@data.csv"
        ```
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.parquet')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only CSV and Parquet files are supported."
        )
    
    temp_dir = Path("/tmp") / "data_science_agent"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = None
    
    try:
        # Save file temporarily
        temp_file_path = temp_dir / file.filename
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Import profiling tool directly
        from tools.data_profiling import profile_dataset as profile_tool
        from tools.data_profiling import detect_data_quality_issues
        
        # Run profiling tools
        logger.info(f"Profiling dataset: {file.filename}")
        profile_result = profile_tool(str(temp_file_path))
        quality_result = detect_data_quality_issues(str(temp_file_path))
        
        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "profile": profile_result,
                "quality_issues": quality_result
            },
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Profiling failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
    
    finally:
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


@app.get("/tools")
async def list_tools():
    """
    List all available tools in the agent.
    
    Returns tool names organized by category.
    Useful for understanding agent capabilities.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    from tools.tools_registry import get_tools_by_category
    
    return {
        "total_tools": len(agent.tool_functions),
        "tools_by_category": get_tools_by_category(),
        "all_tools": list(agent.tool_functions.keys())
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom error response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all error handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
            "error_type": type(exc).__name__
        }
    )


# ==================== STATIC FILE SERVING ====================
# Serve React frontend build in production
frontend_build_path = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_build_path.exists():
    # Serve static assets
    app.mount("/assets", StaticFiles(directory=frontend_build_path / "assets"), name="assets")
    
    # Serve index.html for SPA routing
    from fastapi.responses import FileResponse
    
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve React SPA - catch-all route."""
        # Don't serve for API routes
        if full_path.startswith("api/") or full_path in ["health", "tools", "run", "profile"]:
            raise HTTPException(status_code=404, detail="Not found")
        
        index_file = frontend_build_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend not built")


# Cloud Run listens on PORT environment variable
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
