#!/bin/bash

# launch-frontend.sh
# Launch Next.js frontend with configurable port from environment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables if .env exists
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    source "$PROJECT_ROOT/.env"
fi

# Get frontend port from environment or default
FRONTEND_PORT=${APP_API__SERVICES__FRONTEND__PORT:-9000}

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

# Process management
cleanup() {
    log_info "Shutting down frontend..."
    
    if [[ -n "${FRONTEND_PID:-}" ]]; then
        kill $FRONTEND_PID 2>/dev/null || true
        log_info "Frontend (PID: $FRONTEND_PID) terminated"
    fi
    
    # Kill any remaining processes on our port
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    
    log_success "Frontend shutdown complete"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Validation
validate_environment() {
    log_info "Validating frontend environment..."
    
    if [[ ! -d "$PROJECT_ROOT/frontend-nextjs" ]]; then
        log_error "Next.js frontend directory not found at $PROJECT_ROOT/frontend-nextjs"
        exit 1
    fi
    
    if [[ ! -f "$PROJECT_ROOT/frontend-nextjs/package.json" ]]; then
        log_error "package.json not found in Next.js frontend directory"
        exit 1
    fi
    
    # Check required commands
    for cmd in npm node lsof; do
        if ! command -v $cmd > /dev/null; then
            log_error "Required command '$cmd' not found"
            exit 1
        fi
    done
    
    log_success "Environment validation complete"
}

# Port availability check
check_ports() {
    log_info "Checking port $FRONTEND_PORT availability..."
    
    if lsof -i:$FRONTEND_PORT > /dev/null 2>&1; then
        log_warning "Port $FRONTEND_PORT is already in use. Attempting to free it..."
        lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    log_success "Port $FRONTEND_PORT is available"
}

# Start Next.js frontend
start_frontend() {
    log_info "Starting Next.js frontend on port $FRONTEND_PORT..."
    
    cd "$PROJECT_ROOT/frontend-nextjs"
    
    # Install dependencies if needed
    if [[ ! -d "node_modules" ]]; then
        log_info "Installing Next.js dependencies..."
        npm install
    fi
    
    # Create/update .env.local with port configuration
    echo "PORT=$FRONTEND_PORT" > .env.local
    log_info "Configured Next.js to use port $FRONTEND_PORT"
    
    # Start Next.js development server with explicit PORT
    log_info "Starting Next.js development server..."
    PORT=$FRONTEND_PORT npm run dev > "/tmp/frontend-next.log" 2>&1 &
    FRONTEND_PID=$!
    
    log_info "Next.js frontend started with PID: $FRONTEND_PID"
    
    # Wait for frontend to be ready
    log_info "Waiting for frontend to be ready..."
    sleep 8  # Give Next.js time to compile
    
    # Check if still running
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        log_success "Next.js frontend is running at http://localhost:$FRONTEND_PORT"
    else
        log_error "Frontend failed to start. Check logs at /tmp/frontend-next.log"
        exit 1
    fi
}

# Display service information
show_services() {
    echo ""
    echo "=============================================="
    echo "🚀 Next.js Frontend Service"
    echo "=============================================="
    echo ""
    echo "📡 Service Running:"
    echo "  Frontend:         http://localhost:$FRONTEND_PORT"
    echo ""
    echo "📝 Log Files:"
    echo "  Frontend:         /tmp/frontend-next.log"
    echo ""
    echo "🔧 Configuration:"
    echo "  Port Source:      APP_API__SERVICES__FRONTEND__PORT=${APP_API__SERVICES__FRONTEND__PORT:-'(default)'}"
    echo "  Actual Port:      $FRONTEND_PORT"
    echo ""
    echo "Press Ctrl+C to stop the frontend"
    echo "=============================================="
}

# Main execution
main() {
    log_info "Starting Next.js frontend service..."
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)
                FRONTEND_PORT="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --port PORT    Frontend port [default: from APP_API__SERVICES__FRONTEND__PORT or 9000]"
                echo "  --help         Show this help message"
                echo ""
                echo "Environment Variables:"
                echo "  APP_API__SERVICES__FRONTEND__PORT    Set frontend port (overrides default)"
                echo ""
                echo "Examples:"
                echo "  $0                      # Use port from environment or default (9000)"
                echo "  $0 --port 3000         # Use specific port"
                echo "  APP_API__SERVICES__FRONTEND__PORT=8080 $0  # Use environment variable"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Validate and start frontend
    validate_environment
    check_ports
    start_frontend
    show_services
    
    # Wait for process
    wait $FRONTEND_PID
}

# Run main function
main "$@"