#!/usr/bin/env python3
"""Development deployment orchestration for Phase 5."""

import asyncio
import signal
import sys
import subprocess
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DevDeploymentManager:
    """Manage development deployment lifecycle."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.processes: Dict[str, subprocess.Popen] = {}
        self.shutdown_requested = False
        
        # Service configurations
        self.services = {
            "calendar_api": {
                "command": ["uv", "run", "python", "-m", "services.calendar_api.main"],
                "port": 9001,
                "description": "Calendar API Service"
            },
            "equity_api": {
                "command": ["uv", "run", "python", "-m", "services.equity_api.main"],
                "port": 9002,
                "description": "Equity API Service"
            },
            "option_chain_api": {
                "command": ["uv", "run", "python", "-m", "services.option_chain_api.main"],
                "port": 9004,
                "description": "Option Chain API Service"
            },
            "orchestrator_api": {
                "command": ["uv", "run", "python", "-m", "services.orchestrator_api.main"],
                "port": 9005,
                "description": "Orchestrator API Service"
            },
        }
        
        # Infrastructure services
        self.infrastructure = {
            "redis": {
                "check_command": ["redis-cli", "ping"],
                "start_command": None,  # Assume Redis is already running
                "port": 6379,
                "description": "Redis (Celery Broker)"
            }
        }
        
        # Worker services
        self.workers = {
            "celery_worker": {
                "command": ["uv", "run", "celery", "-A", "ops.celery_app", "worker", "--loglevel=info", "--concurrency=2"],
                "description": "Celery Worker"
            },
            "celery_beat": {
                "command": ["uv", "run", "celery", "-A", "ops.celery_app", "beat", "--loglevel=info"],
                "description": "Celery Beat Scheduler"
            }
        }
        
        # Monitoring services
        self.monitoring = {
            "flower": {
                "command": ["uv", "run", "celery", "-A", "ops.celery_app", "flower", "--port=5555"],
                "port": 5555,
                "description": "Flower (Celery Monitoring)"
            }
        }
    
    async def start_development_stack(self):
        """Start complete development environment."""
        try:
            print("🚀 Starting development stack...")
            print("="*60)
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start services in dependency order
            await self._start_infrastructure()
            await self._start_core_services()
            await self._start_workers()
            await self._start_monitoring()
            
            print("\n" + "="*60)
            print("🎉 Development stack started successfully!")
            print("="*60)
            print("📊 Service URLs:")
            print("   • Calendar API:    http://localhost:9001")
            print("   • Equity API:      http://localhost:9002")
            print("   • Option Chain API: http://localhost:9004")
            print("   • Orchestrator API: http://localhost:9005")
            print("   • Flower Monitor:  http://localhost:5555")
            print("="*60)
            print("\n💡 Commands:")
            print("   • View logs: uv run python tools/dev_deployment.py logs <service>")
            print("   • Restart service: uv run python tools/dev_deployment.py restart <service>")
            print("   • Stop all: Ctrl+C")
            print("="*60)
            
            # Keep running until shutdown
            await self._wait_for_shutdown()
            
        except Exception as e:
            logger.error(f"Failed to start development stack: {e}")
            await self.shutdown_all()
            sys.exit(1)
    
    async def _start_infrastructure(self):
        """Start infrastructure services."""
        print("\n🔧 Starting infrastructure services...")
        
        # Check if Redis is running
        if not self._is_redis_running():
            print("⚠️  Redis not detected. Please start Redis manually:")
            print("   • macOS: brew services start redis")
            print("   • Linux: sudo systemctl start redis")
            print("   • Docker: docker run -d -p 6379:6379 redis:alpine")
            print("\nWaiting for Redis...")
            while not self._is_redis_running():
                await asyncio.sleep(2)
        
        print("✅ Redis is running")
    
    async def _start_core_services(self):
        """Start core API services."""
        print("\n🌐 Starting core API services...")
        
        for service_name, config in self.services.items():
            await self._start_service(service_name, config)
            
            # Wait for service to be ready
            if "port" in config:
                await self._wait_for_port(config["port"], service_name)
    
    async def _start_workers(self):
        """Start Celery workers."""
        print("\n👷 Starting worker services...")
        
        for service_name, config in self.workers.items():
            await self._start_service(service_name, config)
            # Small delay between worker starts
            await asyncio.sleep(2)
    
    async def _start_monitoring(self):
        """Start development monitoring tools."""
        print("\n📊 Starting monitoring services...")
        
        for service_name, config in self.monitoring.items():
            await self._start_service(service_name, config)
            
            if "port" in config:
                await self._wait_for_port(config["port"], service_name)
    
    async def _start_service(self, service_name: str, config: Dict):
        """Start a single service."""
        if service_name in self.processes:
            logger.warning(f"Service {service_name} is already running")
            return
        
        try:
            logger.info(f"Starting {config['description']}...")
            
            process = subprocess.Popen(
                config["command"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[service_name] = process
            print(f"   ✅ Started {config['description']} (PID: {process.pid})")
            
        except Exception as e:
            logger.error(f"Failed to start {service_name}: {e}")
    
    async def _wait_for_port(self, port: int, service_name: str, timeout: int = 30):
        """Wait for a service to be available on a port."""
        import socket
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    logger.info(f"   ✅ {service_name} is ready on port {port}")
                    return True
            except (socket.error, ConnectionRefusedError):
                await asyncio.sleep(1)
        
        logger.warning(f"   ⚠️  {service_name} not responding on port {port} after {timeout}s")
        return False
    
    def _is_redis_running(self) -> bool:
        """Check if Redis is running."""
        try:
            result = subprocess.run(
                ["redis-cli", "ping"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0 and "PONG" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}. Shutting down...")
        self.shutdown_requested = True
    
    async def _wait_for_shutdown(self):
        """Wait for shutdown signal or process failures."""
        while not self.shutdown_requested:
            # Check if any process has died
            dead_services = []
            for service_name, process in self.processes.items():
                if process.poll() is not None:  # Process has terminated
                    dead_services.append(service_name)
            
            if dead_services:
                for service_name in dead_services:
                    process = self.processes.pop(service_name)
                    logger.error(f"Service {service_name} died (exit code: {process.returncode})")
                    
                    # Show stderr if available
                    if process.stderr:
                        stderr = process.stderr.read()
                        if stderr:
                            logger.error(f"Service {service_name} stderr: {stderr}")
            
            await asyncio.sleep(2)
        
        await self.shutdown_all()
    
    async def shutdown_all(self):
        """Shutdown all services gracefully."""
        if not self.processes:
            return
        
        print("\n🛑 Shutting down all services...")
        
        # Send SIGTERM to all processes
        for service_name, process in self.processes.items():
            try:
                logger.info(f"Stopping {service_name}...")
                process.terminate()
            except ProcessLookupError:
                pass  # Process already dead
        
        # Wait for graceful shutdown
        await asyncio.sleep(3)
        
        # Force kill any remaining processes
        for service_name, process in list(self.processes.items()):
            try:
                if process.poll() is None:  # Still running
                    logger.warning(f"Force killing {service_name}...")
                    process.kill()
                process.wait(timeout=5)
                print(f"   ✅ Stopped {service_name}")
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass
        
        self.processes.clear()
        print("🏁 All services stopped")
    
    async def restart_service(self, service_name: str):
        """Restart a specific service."""
        # Find service config
        config = None
        if service_name in self.services:
            config = self.services[service_name]
        elif service_name in self.workers:
            config = self.workers[service_name] 
        elif service_name in self.monitoring:
            config = self.monitoring[service_name]
        
        if not config:
            logger.error(f"Unknown service: {service_name}")
            return
        
        # Stop existing process
        if service_name in self.processes:
            process = self.processes.pop(service_name)
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        # Start new process
        await self._start_service(service_name, config)
        logger.info(f"✅ Restarted {service_name}")
    
    async def get_service_status(self, service_name: str) -> str:
        """Get status of a specific service."""
        if service_name not in self.processes:
            return "stopped"
        
        process = self.processes[service_name]
        if process.poll() is None:
            return "running"
        else:
            return f"dead (exit code: {process.returncode})"
    
    async def get_all_service_statuses(self) -> Dict[str, str]:
        """Get status of all services."""
        statuses = {}
        
        all_services = {**self.services, **self.workers, **self.monitoring}
        for service_name in all_services:
            statuses[service_name] = await self.get_service_status(service_name)
        
        return statuses


async def main():
    """Main entry point for development deployment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Development Deployment Manager")
    parser.add_argument("command", choices=["start", "stop", "status", "restart"], 
                       help="Command to execute")
    parser.add_argument("service", nargs="?", help="Service name (for restart/status)")
    
    args = parser.parse_args()
    
    manager = DevDeploymentManager()
    
    if args.command == "start":
        await manager.start_development_stack()
    elif args.command == "stop":
        await manager.shutdown_all()
    elif args.command == "status":
        if args.service:
            status = await manager.get_service_status(args.service)
            print(f"{args.service}: {status}")
        else:
            statuses = await manager.get_all_service_statuses()
            print("Service Status:")
            for service, status in statuses.items():
                print(f"  {service}: {status}")
    elif args.command == "restart":
        if not args.service:
            print("Service name required for restart")
            sys.exit(1)
        await manager.restart_service(args.service)


if __name__ == "__main__":
    asyncio.run(main())