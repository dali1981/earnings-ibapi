#!/bin/bash

# create-launch-script.sh
# Generate launch scripts for services based on the template

set -eo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Service configurations (service:module:port)
get_service_config() {
    local service=$1
    case $service in
        "equity") echo "services.equity_api.app:app:9002" ;;
        "options") echo "services.option_history_api.app:app:9003" ;;
        "chains") echo "api.option_chain_api.app:app:9004" ;;
        "orchestrator") echo "api.orchestrator_api.app:app:9005" ;;
        "ib") echo "api.ib_api.app:app:9010" ;;
        "trading") echo "services.trading_api.app:app:9011" ;;
        *) echo "" ;;
    esac
}

# List of available services
AVAILABLE_SERVICES="equity options chains orchestrator ib trading"

# Function to create service-specific endpoints info
get_service_endpoints() {
    local service=$1
    local port=$2
    
    case $service in
        "equity")
            cat << EOF
    echo "📈 Key Equity Endpoints:"
    echo "  Available Dates:  http://localhost:$port/equity/{symbol}/dates"
    echo "  Historical Bars:  http://localhost:$port/equity/{symbol}/bars"
    echo "  Storage Info:     http://localhost:$port/equity/storage-info"
EOF
            ;;
        "trading")
            cat << EOF
    echo "💰 Key Trading Endpoints:"
    echo "  Create Order:     http://localhost:$port/orders"
    echo "  Positions:        http://localhost:$port/positions"
    echo "  Portfolio:        http://localhost:$port/portfolio/summary"
    echo "  Risk Assessment:  http://localhost:$port/risk/assess"
    echo "  Strategy Status:  http://localhost:$port/strategies/status"
EOF
            ;;
        "orchestrator")
            cat << EOF
    echo "🎯 Key Orchestrator Endpoints:"
    echo "  Contract Search:  http://localhost:$port/orchestrator/contracts/search"
    echo "  Historical Data:  http://localhost:$port/orchestrator/historical/bars"
    echo "  Task Status:      http://localhost:$port/orchestrator/tasks/status"
EOF
            ;;
        "options")
            cat << EOF
    echo "📊 Key Options Endpoints:"
    echo "  Available Dates:  http://localhost:$port/options/{symbol}/dates"
    echo "  Historical Data:  http://localhost:$port/options/{symbol}/bars"
    echo "  Storage Stats:    http://localhost:$port/options/stats"
EOF
            ;;
        "chains")
            cat << EOF
    echo "⛓️  Key Chain Endpoints:"
    echo "  Current Chains:   http://localhost:$port/chains/current/{symbol}"
    echo "  Snapshots:        http://localhost:$port/chains/snapshots"
    echo "  Chain Stats:      http://localhost:$port/chains/stats"
EOF
            ;;
        "ib")
            cat << EOF
    echo "🏦 Key IB API Endpoints:"
    echo "  Connection:       http://localhost:$port/ib/connection/status"
    echo "  Market Data:      http://localhost:$port/ib/market-data"
    echo "  Contract Details: http://localhost:$port/ib/contracts"
EOF
            ;;
        *)
            cat << EOF
    echo "📈 Key $service Endpoints:"
    echo "  Service Health:   http://localhost:$port/health"
    echo "  API Documentation: http://localhost:$port/docs"
EOF
            ;;
    esac
}

# Generate launch script for a service
generate_script() {
    local service_name=$1
    local service_module=$2
    local service_port=$3
    
    local script_path="$SCRIPT_DIR/launch-${service_name}.sh"
    
    log_info "Generating launch script for $service_name service..."
    
    # Copy template and customize
    cp "$SCRIPT_DIR/launch-template.sh" "$script_path"
    
    # Replace template variables
    sed -i '' "s/SERVICE_NAME=\"template\"/SERVICE_NAME=\"$service_name\"/g" "$script_path"
    sed -i '' "s|SERVICE_MODULE=\"services.template_api.app:app\"|SERVICE_MODULE=\"$service_module\"|g" "$script_path"
    sed -i '' "s/SERVICE_PORT=9000/SERVICE_PORT=$service_port/g" "$script_path"
    
    # Create temporary file with service-specific endpoints
    local temp_file="/tmp/service_endpoints_$$.txt"
    get_service_endpoints "$service_name" "$service_port" > "$temp_file"
    
    # Insert service-specific endpoints using awk (more compatible than sed)
    awk -v endpoints_file="$temp_file" '
        /📝 Log Files:/ {
            while ((getline line < endpoints_file) > 0) {
                print "    " line
            }
            close(endpoints_file)
            print "    echo \"\""
        }
        { print }
    ' "$script_path" > "${script_path}.tmp" && mv "${script_path}.tmp" "$script_path"
    
    # Clean up temp file
    rm -f "$temp_file"
    
    # Make executable
    chmod +x "$script_path"
    
    log_success "Created launch script: $script_path"
}

# Show usage
show_usage() {
    echo "Usage: $0 [SERVICE_NAME] | --all | --list"
    echo ""
    echo "Generate launch scripts for trading project services."
    echo ""
    echo "Options:"
    echo "  SERVICE_NAME    Generate script for specific service"
    echo "  --all           Generate scripts for all services"  
    echo "  --list          List available services"
    echo "  --help          Show this help message"
    echo ""
    echo "Available services:"
    for service in $AVAILABLE_SERVICES; do
        local config=$(get_service_config "$service")
        if [[ -n "$config" ]]; then
            local port=$(echo "$config" | cut -d: -f3)
            echo "  $service (port $port)"
        fi
    done
    echo ""
    echo "Examples:"
    echo "  $0 equity           # Generate launch script for equity service"
    echo "  $0 --all            # Generate scripts for all services"
    echo "  $0 --list           # Show available services"
}

# Main execution
main() {
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 1
    fi
    
    case $1 in
        --help)
            show_usage
            exit 0
            ;;
        --list)
            echo "Available services:"
            for service in $AVAILABLE_SERVICES; do
                local config=$(get_service_config "$service")
                if [[ -n "$config" ]]; then
                    local module=$(echo "$config" | cut -d: -f1-2)
                    local port=$(echo "$config" | cut -d: -f3)
                    echo "  $service (module: $module, port: $port)"
                fi
            done
            exit 0
            ;;
        --all)
            log_info "Generating launch scripts for all services..."
            for service in $AVAILABLE_SERVICES; do
                local config=$(get_service_config "$service")
                if [[ -n "$config" ]]; then
                    local module=$(echo "$config" | cut -d: -f1-2)
                    local port=$(echo "$config" | cut -d: -f3)
                    generate_script "$service" "$module" "$port"
                fi
            done
            log_success "Generated launch scripts for all services"
            ;;
        *)
            service_name=$1
            local config=$(get_service_config "$service_name")
            if [[ -z "$config" ]]; then
                log_error "Unknown service: $service_name"
                echo ""
                show_usage
                exit 1
            fi
            
            local module=$(echo "$config" | cut -d: -f1-2)
            local port=$(echo "$config" | cut -d: -f3)
            generate_script "$service_name" "$module" "$port"
            ;;
    esac
}

main "$@"