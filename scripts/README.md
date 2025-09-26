# Trading Project Launch Scripts

This directory contains launch scripts for running Trading Project services with integrated frontend support.

## 🚀 Quick Start

### Launch Calendar API with Frontend
```bash
# Start with Next.js frontend (default)
./scripts/launch-calendar.sh

# Start with React frontend
./scripts/launch-calendar.sh --frontend react

# Start with legacy FastAPI frontend
./scripts/launch-calendar.sh --frontend legacy

# Custom ports
./scripts/launch-calendar.sh --calendar-port 9001 --frontend-port 3000
```

## 📁 Available Scripts

### Core Scripts
- `launch-calendar.sh` - Calendar API with frontend integration
- `launch-template.sh` - Template for creating new service scripts
- `create-launch-script.sh` - Generate launch scripts for other services

### Features
- 🔄 **Auto-reload** for both backend and frontend during development
- 🏥 **Health checks** to ensure services start correctly
- 🧹 **Cleanup** on exit (Ctrl+C) - kills all spawned processes
- 📝 **Comprehensive logging** with colored output
- 🔧 **Configurable ports** and frontend types
- ⚡ **Fast startup** with dependency validation

## 🛠️ Generate Scripts for Other Services

Use the script generator to create launch scripts for other services:

```bash
# Generate script for a specific service
./scripts/create-launch-script.sh equity

# Generate scripts for all services
./scripts/create-launch-script.sh --all

# List available services
./scripts/create-launch-script.sh --list
```

### Available Services
- `equity` (port 9002) - Equity historical data API
- `options` (port 9003) - Options historical data API  
- `chains` (port 9004) - Option chain snapshots API
- `orchestrator` (port 9005) - High-level workflow orchestration
- `ib` (port 9010) - Interactive Brokers API integration
- `trading` (port 9011) - Live trading operations API

## 🖥️ Frontend Options

### Next.js Frontend (Recommended)
- **Path**: `frontend-nextjs/`
- **Tech**: Next.js 15.5.2 + React 19 + TypeScript + shadcn/ui
- **Port**: 3000 (default)
- **Command**: `--frontend next`

### React Frontend  
- **Path**: `frontend-react/`
- **Tech**: React + Vite + Custom CSS
- **Port**: 3000
- **Command**: `--frontend react`

### Legacy FastAPI Frontend
- **Path**: `frontend/`
- **Tech**: FastAPI + Jinja2 templates
- **Port**: 9000 (configurable)
- **Command**: `--frontend legacy`

## 📊 Service Information Display

Each launch script provides a comprehensive dashboard showing:

```
============================================== 
🚀 Calendar API Service Stack
==============================================

📡 Services Running:
  Calendar API:     http://localhost:9001
  API Docs:         http://localhost:9001/docs
  Frontend (next):  http://localhost:3000

📊 Health Endpoints:
  Calendar Health:  http://localhost:9001/health
  Detailed Health:  http://localhost:9001/calendar/health-detailed

📈 Key Calendar Endpoints:
  Upcoming Events:  http://localhost:9001/calendar/upcoming
  Statistics:       http://localhost:9001/calendar/stats
  Data Availability: http://localhost:9001/calendar/data-availability

📝 Log Files:
  Calendar API:     /tmp/calendar-api.log
  Frontend:         /tmp/frontend-next.log

Press Ctrl+C to stop all services
==============================================
```

## 🔧 Script Customization

### Creating New Service Scripts

1. **Use the generator** (recommended):
   ```bash
   ./scripts/create-launch-script.sh your-service
   ```

2. **Manual customization** of `launch-template.sh`:
   ```bash
   cp scripts/launch-template.sh scripts/launch-your-service.sh
   ```
   
   Then edit these variables:
   ```bash
   SERVICE_NAME="your-service"
   SERVICE_MODULE="services.your_service_api.app:app"
   SERVICE_PORT=9012
   SERVICE_HEALTH_ENDPOINT="/health"
   ```

### Adding Service-Specific Endpoints

Customize the `show_services()` function to display relevant endpoints for your service:

```bash
echo "📈 Key Your-Service Endpoints:"
echo "  Main Feature:     http://localhost:$SERVICE_PORT/your-endpoint"
echo "  Analytics:        http://localhost:$SERVICE_PORT/your-stats"
```

## 🏥 Health Checks & Monitoring

All scripts include:
- **Startup validation** - Ensures services start correctly
- **Port conflict resolution** - Automatically frees busy ports
- **Process monitoring** - Tracks PIDs for clean shutdown
- **Comprehensive logging** - Separate log files for each service

### Log File Locations
- Backend services: `/tmp/{service-name}-api.log`
- Next.js frontend: `/tmp/frontend-next.log`
- React frontend: `/tmp/frontend-react.log`
- Legacy frontend: `/tmp/frontend-legacy.log`

## 🔄 Integration with Existing Workflow

These scripts complement the existing Makefile targets:

```bash
# Existing single service commands
make run-calendar    # Calendar only
make run-frontend    # Frontend only

# New integrated commands  
./scripts/launch-calendar.sh    # Calendar + Frontend together
```

### Backend Integration
The scripts work seamlessly with:
- `make run-backend` - All backend services
- `make services-up` - Docker compose stack
- `make celery-worker` - Background task processing

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Scripts automatically handle this, but manual cleanup:
   lsof -ti:9001 | xargs kill -9
   ```

2. **Frontend dependencies missing**
   ```bash
   # Scripts auto-install, but manual install:
   cd frontend-nextjs && npm install
   ```

3. **Virtual environment not activated**
   ```bash
   # Scripts handle this automatically
   source venv/bin/activate
   ```

4. **Service fails to start**
   ```bash
   # Check log files
   tail -f /tmp/calendar-api.log
   tail -f /tmp/frontend-next.log
   ```

### Debug Mode
Run scripts with verbose logging:
```bash
set -x  # Add to script for debug output
./scripts/launch-calendar.sh --frontend next
```

## 📈 Performance Considerations

- **Development mode**: All services run with `--reload` for fast iteration
- **Log rotation**: Log files are overwritten on each run
- **Memory usage**: Each frontend type has different resource requirements
- **Startup time**: Next.js takes longer to compile than React or legacy

## 🔮 Future Enhancements

Planned improvements:
- **Docker integration** - Containerized service stacks
- **Load balancing** - Multiple service instances  
- **Service mesh** - Advanced inter-service communication
- **Monitoring dashboard** - Centralized health and metrics
- **Production mode** - Optimized builds and configurations