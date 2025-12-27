#!/bin/bash
# Development script - runs both frontend and backend concurrently

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Data Science Agent - Development Mode${NC}"
echo "=========================================="
echo ""

# Check if required tools are installed
check_requirements() {
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Python not found. Please install Python 3.11+${NC}"
        exit 1
    fi
    
    if ! command -v node &> /dev/null; then
        echo -e "${RED}❌ Node.js not found. Please install Node.js 18+${NC}"
        exit 1
    fi
    
    if ! command -v npm &> /dev/null; then
        echo -e "${RED}❌ npm not found. Please install npm${NC}"
        exit 1
    fi
}

# Setup Python virtual environment
setup_python() {
    echo -e "${YELLOW}📦 Setting up Python environment...${NC}"
    
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        echo -e "${GREEN}✅ Created virtual environment${NC}"
    fi
    
    source .venv/bin/activate
    
    # Install dependencies if needed
    if ! pip show fastapi &> /dev/null; then
        echo -e "${YELLOW}📥 Installing Python dependencies...${NC}"
        pip install -r requirements.txt
    fi
    
    echo -e "${GREEN}✅ Python environment ready${NC}"
}

# Setup Node.js dependencies
setup_node() {
    echo -e "${YELLOW}📦 Setting up Node.js environment...${NC}"
    
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}📥 Installing Node.js dependencies...${NC}"
        npm install
    fi
    
    cd ..
    
    echo -e "${GREEN}✅ Node.js environment ready${NC}"
}

# Start the backend
start_backend() {
    echo -e "${BLUE}🔧 Starting FastAPI backend on port 8080...${NC}"
    
    source .venv/bin/activate
    cd src/api
    python app.py &
    BACKEND_PID=$!
    cd ../..
    
    echo -e "${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"
}

# Start the frontend
start_frontend() {
    echo -e "${BLUE}🎨 Starting React frontend on port 3000...${NC}"
    
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..
    
    echo -e "${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}"
}

# Cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down...${NC}"
    
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        echo "  Stopped backend"
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo "  Stopped frontend"
    fi
    
    # Kill any remaining processes on the ports
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    
    echo -e "${GREEN}✅ Shutdown complete${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main
main() {
    check_requirements
    setup_python
    setup_node
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}🚀 Starting development servers...${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    start_backend
    
    # Wait for backend to start
    sleep 3
    
    start_frontend
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Development environment ready!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "🎨 Frontend:  ${BLUE}http://localhost:3000${NC}"
    echo -e "🔧 Backend:   ${BLUE}http://localhost:8080${NC}"
    echo -e "📚 API Docs:  ${BLUE}http://localhost:8080/docs${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
    echo ""
    
    # Wait for both processes
    wait
}

main
