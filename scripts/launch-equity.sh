#!/bin/bash

# launch-template.sh
# Template for creating service launch scripts
# Copy this file and customize for each service

set -euo pipefail

# Service Configuration - CUSTOMIZE THESE FOR EACH SERVICE
SERVICE_NAME="equity"           # e.g., "calendar", "equity", "trading"
SERVICE_MODULE="services.equity_api.app:app"  # e.g., "services.calendar_api.app:app"
SERVICE_PORT=9002                 # Default port for this service
SERVICE_HEALTH_ENDPOINT="/health" # Health check endpoint

# Frontend Configuration
FRONTEND_PORT=${APP_API__SERVICES__FRONTEND__PORT:-9000}
FRONTEND_TYPE="next"              # Options: next, react

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Process management functions
cleanup() {
    log_info "Shutting down $SERVICE_NAME API and Frontend..."
    
    # Kill background processes if they exist
    if [[ -n "${SERVICE_PID:-}" ]]; then
        kill $SERVICE_PID 2>/dev/null || true
        log_info "$SERVICE_NAME API (PID: $SERVICE_PID) terminated"
    fi
    
    if [[ -n "${FRONTEND_PID:-}" ]]; then
        kill $FRONTEND_PID 2>/dev/null || true
        log_info "Frontend (PID: $FRONTEND_PID) terminated"
    fi
    
    # Kill any remaining processes on our ports
    lsof -ti:$SERVICE_PORT | xargs kill -9 2>/dev/null || true
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    
    log_success "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Health check function
check_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=0
    
    log_info "Waiting for $service_name to be ready..."
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            log_success "$service_name is ready at $url"
            return 0
        fi
        
        ((attempt++))
        sleep 1
        echo -n "."
    done
    
    log_error "$service_name failed to start within $max_attempts seconds"
    return 1
}

# Validation functions
validate_environment() {
    log_info "Validating environment for $SERVICE_NAME service..."
    
    # Check if we're in the right directory
    SERVICE_APP_PATH="$(echo $SERVICE_MODULE | cut -d: -f1 | tr . /)"
    if [[ ! -f "$PROJECT_ROOT/${SERVICE_APP_PATH}.py" ]]; then
        log_error "$SERVICE_NAME API app.py not found at $PROJECT_ROOT/${SERVICE_APP_PATH}.py"
        log_error "Are you running from the correct directory?"
        exit 1
    fi
    
    # Check Python virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
            log_info "Activating Python virtual environment..."
            source "$PROJECT_ROOT/venv/bin/activate"
        else
            log_warning "No virtual environment detected. Using system Python."
        fi
    fi
    
    # Check required commands
    for cmd in uvicorn curl lsof; do
        if ! command -v $cmd > /dev/null; then
            log_error "Required command '$cmd' not found"
            exit 1
        fi
    done
    
    # Check frontend directory
    case $FRONTEND_TYPE in
        "next")
            if [[ ! -d "$PROJECT_ROOT/frontend-nextjs" ]]; then
                log_error "Next.js frontend directory not found"
                exit 1
            fi
            ;;
        "react")
            if [[ ! -d "$PROJECT_ROOT/frontend-react" ]]; then
                log_error "React frontend directory not found"
                exit 1
            fi
            ;;
        "legacy")
            if [[ ! -f "$PROJECT_ROOT/frontend/main.py" ]]; then
                log_error "Legacy frontend not found"
                exit 1
            fi
            ;;
    esac
    
    log_success "Environment validation complete"
}

# Port availability check
check_ports() {
    log_info "Checking port availability..."
    
    if lsof -i:$SERVICE_PORT > /dev/null 2>&1; then
        log_warning "Port $SERVICE_PORT is already in use. Attempting to free it..."
        lsof -ti:$SERVICE_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    if lsof -i:$FRONTEND_PORT > /dev/null 2>&1; then
        log_warning "Port $FRONTEND_PORT is already in use. Attempting to free it..."
        lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    log_success "Ports are available"
}

# Start Service API
start_service_api() {
    log_info "Starting $SERVICE_NAME API on port $SERVICE_PORT..."
    
    cd "$PROJECT_ROOT"
    
    # Start service API in background
    uvicorn $SERVICE_MODULE \
        --host 0.0.0.0 \
        --port $SERVICE_PORT \
        --reload \
        --reload-dir "$(dirname "$(echo $SERVICE_MODULE | cut -d: -f1 | tr . /)")" \
        --log-level info > "/tmp/${SERVICE_NAME}-api.log" 2>&1 &
    
    SERVICE_PID=$!
    log_info "$SERVICE_NAME API started with PID: $SERVICE_PID"
    
    # Wait for service to be ready
    if check_service "http://localhost:$SERVICE_PORT$SERVICE_HEALTH_ENDPOINT" "$SERVICE_NAME API"; then
        log_success "$SERVICE_NAME API is running at http://localhost:$SERVICE_PORT"
        log_info "API documentation: http://localhost:$SERVICE_PORT/docs"
    else
        log_error "$SERVICE_NAME API failed to start"
        cleanup
        exit 1
    fi
}

# Start Frontend
start_frontend() {
    log_info "Starting $FRONTEND_TYPE frontend on port $FRONTEND_PORT..."
    
    case $FRONTEND_TYPE in
        "next")
            cd "$PROJECT_ROOT/frontend-nextjs"
            
            # Install dependencies if needed
            if [[ ! -d "node_modules" ]]; then
                log_info "Installing Next.js dependencies..."
                npm install
            fi
            
            # Start Next.js development server with explicit PORT
            PORT=$FRONTEND_PORT npm run dev > "/tmp/frontend-next.log" 2>&1 &
            FRONTEND_PID=$!
            ;;
            
        "react")
            cd "$PROJECT_ROOT/frontend-react"
            
            # Install dependencies if needed
            if [[ ! -d "node_modules" ]]; then
                log_info "Installing React dependencies..."
                npm install
            fi
            
            # Start React development server
            npm run dev > "/tmp/frontend-react.log" 2>&1 &
            FRONTEND_PID=$!
            ;;
            
        "legacy")
            cd "$PROJECT_ROOT"
            
            # Start legacy FastAPI frontend
            uvicorn frontend.main:app \
                --host 0.0.0.0 \
                --port $FRONTEND_PORT \
                --reload > "/tmp/frontend-legacy.log" 2>&1 &
            FRONTEND_PID=$!
            ;;
    esac
    
    log_info "Frontend ($FRONTEND_TYPE) started with PID: $FRONTEND_PID"
    
    # Wait for frontend to be ready
    sleep 5  # Give frontend time to compile/start
    log_success "Frontend is running at http://localhost:$FRONTEND_PORT"
}

# Display service information - CUSTOMIZE FOR EACH SERVICE
show_services() {
    echo ""
    echo "=============================================="
    echo "🚀 $SERVICE_NAME API Service Stack"
    echo "=============================================="
    echo ""
    echo "📡 Services Running:"
    echo "  $SERVICE_NAME API:       http://localhost:$SERVICE_PORT"
    echo "  API Docs:         http://localhost:$SERVICE_PORT/docs"
    echo "  Frontend ($FRONTEND_TYPE):     http://localhost:$FRONTEND_PORT"
    echo ""
    echo "📊 Health Endpoints:"
    echo "  Service Health:   http://localhost:$SERVICE_PORT$SERVICE_HEALTH_ENDPOINT"
    echo ""
        echo "📈 Key Equity Endpoints:"
        echo "  Available Dates:  http://localhost:9002/equity/{symbol}/dates"
        echo "  Historical Bars:  http://localhost:9002/equity/{symbol}/bars"
        echo "  Storage Info:     http://localhost:9002/equity/storage-info"
    echo ""
    echo "📝 Log Files:"
    echo "  $SERVICE_NAME API:       /tmp/${SERVICE_NAME}-api.log"
    echo "  Frontend:         /tmp/frontend-$FRONTEND_TYPE.log"
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo "=============================================="
}

# Main execution
main() {
    log_info "Starting $SERVICE_NAME API service stack..."
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --frontend)
                FRONTEND_TYPE="$2"
                shift 2
                ;;
            --service-port)
                SERVICE_PORT="$2"
                shift 2
                ;;
            --frontend-port)
                FRONTEND_PORT="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --frontend TYPE       Frontend type (next, react, legacy) [default: next]"
                echo "  --service-port PORT   $SERVICE_NAME API port [default: $SERVICE_PORT]"
                echo "  --frontend-port PORT  Frontend port [default: $FRONTEND_PORT]"
                echo "  --help               Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                                    # Start with Next.js frontend"
                echo "  $0 --frontend react                  # Start with React frontend"
                echo "  $0 --frontend legacy                 # Start with legacy FastAPI frontend"
                echo "  $0 --service-port $SERVICE_PORT --frontend-port $FRONTEND_PORT"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Validate and start services
    validate_environment
    check_ports
    start_service_api
    start_frontend
    show_services
    
    # Wait for processes
    wait
}

# Run main function
main "$@"
