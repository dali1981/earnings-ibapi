#!/usr/bin/env python3
"""Development CLI commands for Phase 5."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from tools.dev_deployment import DevDeploymentManager


async def dev_start():
    """Start development environment."""
    manager = DevDeploymentManager()
    await manager.start_development_stack()


async def dev_stop():
    """Stop development environment."""
    manager = DevDeploymentManager()
    await manager.shutdown_all()


async def dev_status():
    """Check development environment status."""
    manager = DevDeploymentManager()
    statuses = await manager.get_all_service_statuses()
    
    print("📊 Development Environment Status:")
    print("="*50)
    for service, status in statuses.items():
        icon = "✅" if status == "running" else "❌" if status.startswith("dead") else "⚪"
        print(f"   {icon} {service}: {status}")
    print("="*50)


async def dev_restart(service: str):
    """Restart a specific service."""
    manager = DevDeploymentManager()
    await manager.restart_service(service)


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Development Environment Manager for Phase 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python tools/dev.py start     # Start all services
  uv run python tools/dev.py stop      # Stop all services  
  uv run python tools/dev.py status    # Check service status
  uv run python tools/dev.py restart calendar_api  # Restart specific service

Services available:
  • calendar_api     - Calendar API (port 9001)
  • equity_api       - Equity API (port 9002)  
  • option_chain_api - Option Chain API (port 9004)
  • orchestrator_api - Orchestrator API (port 9005)
  • celery_worker    - Celery Worker
  • flower           - Flower Monitoring (port 5555)
        """
    )
    
    parser.add_argument("command", choices=["start", "stop", "status", "restart"],
                       help="Development command to execute")
    parser.add_argument("service", nargs="?", 
                       help="Service name (required for restart)")
    
    args = parser.parse_args()
    
    if args.command == "start":
        asyncio.run(dev_start())
    elif args.command == "stop":
        asyncio.run(dev_stop())
    elif args.command == "status":
        asyncio.run(dev_status())
    elif args.command == "restart":
        if not args.service:
            print("❌ Error: Service name required for restart command")
            print("\nAvailable services:")
            print("  calendar_api, equity_api, option_chain_api, orchestrator_api")
            print("  celery_worker, celery_beat, flower")
            sys.exit(1)
        asyncio.run(dev_restart(args.service))


if __name__ == "__main__":
    main()