"""
FastAPI Application for Google Cloud Run
Thin HTTP wrapper around DataScienceCopilot - No logic changes, just API exposure.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
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
    version="1.0.0"
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
