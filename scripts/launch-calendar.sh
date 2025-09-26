#!/bin/bash

# launch-calendar.sh
# Launch Calendar API service with frontend integration
# Part of the Trading Project service launch scripts collection

set -euo pipefail

# Configuration
CALENDAR_API_PORT=9001
FRONTEND_PORT=${APP_API__SERVICES__FRONTEND__PORT:-9000}
FRONTEND_TYPE="next"  # Options: next, react

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
    log_info "Shutting down Calendar API and Frontend..."
    
    # Kill background processes if they exist
    if [[ -n "${CALENDAR_PID:-}" ]]; then
        kill $CALENDAR_PID 2>/dev/null || true
        log_info "Calendar API (PID: $CALENDAR_PID) terminated"
    fi
    
    if [[ -n "${FRONTEND_PID:-}" ]]; then
        kill $FRONTEND_PID 2>/dev/null || true
        log_info "Frontend (PID: $FRONTEND_PID) terminated"
    fi
    
    # Kill any remaining processes on our ports
    lsof -ti:$CALENDAR_API_PORT | xargs kill -9 2>/dev/null || true
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
    log_info "Validating environment..."
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/services/calendar_api/app.py" ]]; then
        log_error "Calendar API app.py not found. Are you running from the correct directory?"
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
    esac
    
    log_success "Environment validation complete"
}

# Port availability check
check_ports() {
    log_info "Checking port availability..."
    
    if lsof -i:$CALENDAR_API_PORT > /dev/null 2>&1; then
        log_warning "Port $CALENDAR_API_PORT is already in use. Attempting to free it..."
        lsof -ti:$CALENDAR_API_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    if lsof -i:$FRONTEND_PORT > /dev/null 2>&1; then
        log_warning "Port $FRONTEND_PORT is already in use. Attempting to free it..."
        lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    log_success "Ports are available"
}

# Start Calendar API
start_calendar_api() {
    log_info "Starting Calendar API on port $CALENDAR_API_PORT..."
    
    cd "$PROJECT_ROOT"
    
    # Start calendar API in background
    uvicorn services.calendar_api.app:app \
        --host 0.0.0.0 \
        --port $CALENDAR_API_PORT \
        --reload \
        --reload-dir services/calendar_api \
        --log-level info > /tmp/calendar-api.log 2>&1 &
    
    CALENDAR_PID=$!
    log_info "Calendar API started with PID: $CALENDAR_PID"
    
    # Wait for service to be ready
    if check_service "http://localhost:$CALENDAR_API_PORT/health" "Calendar API"; then
        log_success "Calendar API is running at http://localhost:$CALENDAR_API_PORT"
        log_info "API documentation: http://localhost:$CALENDAR_API_PORT/docs"
    else
        log_error "Calendar API failed to start"
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
            
            # Configure Next.js port via .env.local
            echo "PORT=$FRONTEND_PORT" > .env.local
            
            # Start Next.js development server with explicit PORT
            PORT=$FRONTEND_PORT npm run dev > /tmp/frontend-next.log 2>&1 &
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
            npm run dev > /tmp/frontend-react.log 2>&1 &
            FRONTEND_PID=$!
            ;;
            
    esac
    
    log_info "Frontend ($FRONTEND_TYPE) started with PID: $FRONTEND_PID"
    
    # Wait for frontend to be ready
    sleep 5  # Give frontend time to compile/start
    log_success "Frontend is running at http://localhost:$FRONTEND_PORT"
}

# Display service information
show_services() {
    echo ""
    echo "=============================================="
    echo "🚀 Calendar API Service Stack"
    echo "=============================================="
    echo ""
    echo "📡 Services Running:"
    echo "  Calendar API:     http://localhost:$CALENDAR_API_PORT"
    echo "  API Docs:         http://localhost:$CALENDAR_API_PORT/docs"
    echo "  Frontend ($FRONTEND_TYPE):     http://localhost:$FRONTEND_PORT"
    echo ""
    echo "📊 Health Endpoints:"
    echo "  Calendar Health:  http://localhost:$CALENDAR_API_PORT/health"
    echo "  Detailed Health:  http://localhost:$CALENDAR_API_PORT/calendar/health-detailed"
    echo ""
    echo "📈 Key Calendar Endpoints:"
    echo "  Upcoming Events:  http://localhost:$CALENDAR_API_PORT/calendar/upcoming"
    echo "  Statistics:       http://localhost:$CALENDAR_API_PORT/calendar/stats"
    echo "  Data Availability: http://localhost:$CALENDAR_API_PORT/calendar/data-availability"
    echo ""
    echo "📝 Log Files:"
    echo "  Calendar API:     /tmp/calendar-api.log"
    echo "  Frontend:         /tmp/frontend-$FRONTEND_TYPE.log"
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo "=============================================="
}

# Main execution
main() {
    log_info "Starting Calendar API service stack..."
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --frontend)
                FRONTEND_TYPE="$2"
                shift 2
                ;;
            --calendar-port)
                CALENDAR_API_PORT="$2"
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
                echo "  --frontend TYPE      Frontend type (next, react) [default: next]"
                echo "  --calendar-port PORT Calendar API port [default: 9001]"
                echo "  --frontend-port PORT Frontend port [default: from APP_API__SERVICES__FRONTEND__PORT or 9000]"
                echo "  --help              Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                                    # Start with Next.js frontend"
                echo "  $0 --frontend react                  # Start with React frontend"
                echo "  $0 --calendar-port 9001 --frontend-port 3000"
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
    start_calendar_api
    start_frontend
    show_services
    
    # Wait for processes
    wait
}

# Run main function
main "$@"