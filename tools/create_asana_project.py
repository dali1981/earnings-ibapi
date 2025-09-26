#!/usr/bin/env python3
"""
Create Asana project for Trading Platform Observability implementation.

This script creates a comprehensive project in Asana with all tasks from the 
observability specification, organized by implementation phases.

Requirements:
    pip install asana

Usage:
    python tools/create_asana_project.py --token YOUR_ASANA_TOKEN
"""

import argparse
import asana
from asana.rest import ApiException
from datetime import datetime, timedelta
import sys


class AsanaObservabilityProject:
    def __init__(self, access_token: str):
        self.configuration = asana.Configuration()
        self.configuration.access_token = access_token
        self.api_client = asana.ApiClient(self.configuration)
        
        # Initialize API instances
        self.projects_api = asana.ProjectsApi(self.api_client)
        self.tasks_api = asana.TasksApi(self.api_client)
        self.users_api = asana.UsersApi(self.api_client)
        self.sections_api = asana.SectionsApi(self.api_client)
        
        # Get workspace
        self.workspace_gid = self._get_workspace()
        
    def _get_workspace(self):
        """Get the first available workspace"""
        try:
            me = self.users_api.get_user("me", {})
            if not me.get("workspaces"):
                raise Exception("No workspaces available")
            return me["workspaces"][0]["gid"]
        except ApiException as e:
            print(f"Error getting workspace: {e}")
            sys.exit(1)
    
    def create_project(self):
        """Create the main observability project"""
        project_data = {
            "data": {
                "name": "🔍 Trading Platform - Observability & Sentry Integration",
                "notes": self._get_project_description(),
                "workspace": self.workspace_gid,
                "team": None,  # Will be assigned to personal workspace
                "color": "blue",
                "layout": "list",
                "privacy_setting": "private_to_members"
            }
        }
        
        try:
            project = self.projects_api.create_project(project_data, {})
            print(f"✅ Created project: {project['name']}")
            print(f"   Project GID: {project['gid']}")
            print(f"   Project URL: {project['permalink_url']}")
            return project["gid"]
        except ApiException as e:
            print(f"❌ Error creating project: {e}")
            sys.exit(1)
    
    def _get_project_description(self):
        """Get the project description"""
        return """
🎯 Implementation of comprehensive observability for our trading platform using Sentry as the primary observability platform.

📋 **Key Objectives:**
• Centralized Error Tracking across 6 microservices
• Distributed Tracing for end-to-end transaction visibility  
• Performance Monitoring for APIs and Celery tasks
• Proactive Alerting for business-critical issues
• Business Intelligence for trading-specific metrics

📊 **Architecture:** 6 FastAPI microservices + Celery background processing + Event-driven data pipeline

⏱️ **Timeline:** 7 weeks total implementation (3 phases)
💰 **Budget:** $80/month Sentry Organization Plan
🎯 **Success Metrics:** <5min MTTD, 99.9% API availability, <0.1% error rate

📖 **Full Specification:** docs/observability/OBSERVABILITY_SPEC.md
"""
    
    def create_sections(self, project_gid):
        """Create project sections for organizing tasks"""
        sections = [
            {
                "name": "📋 Project Overview & Planning",
                "description": "High-level planning and specification documents"
            },
            {
                "name": "🏗️ Phase 1: Foundation Setup (Week 1-2)",
                "description": "Core Sentry SDK integration and basic error tracking"
            },
            {
                "name": "🔗 Phase 2: Distributed Tracing (Week 3-4)", 
                "description": "Cross-service tracing and performance monitoring"
            },
            {
                "name": "📊 Phase 3: Advanced Features (Week 5-6)",
                "description": "Custom metrics, frontend monitoring, and business intelligence"
            },
            {
                "name": "🧪 Phase 4: Testing & Optimization (Week 7)",
                "description": "Load testing, alert tuning, and deployment finalization"
            },
            {
                "name": "📚 Documentation & Training",
                "description": "Knowledge transfer and operational handover"
            }
        ]
        
        created_sections = {}
        for section_data in sections:
            try:
                section_request = {
                    "data": {
                        "name": section_data["name"],
                        "project": project_gid
                    }
                }
                
                section = self.sections_api.create_section_for_project(project_gid, section_request, {})
                created_sections[section_data["name"]] = section["gid"]
                print(f"   ✅ Created section: {section_data['name']}")
                
            except ApiException as e:
                print(f"   ❌ Error creating section {section_data['name']}: {e}")
        
        return created_sections
    
    def create_tasks(self, project_gid, sections):
        """Create all tasks organized by sections"""
        
        # Calculate start dates for each phase
        start_date = datetime.now()
        
        task_groups = [
            # Project Overview & Planning
            {
                "section": "📋 Project Overview & Planning",
                "tasks": [
                    {
                        "name": "📖 Review Observability Specification",
                        "notes": "Review docs/observability/OBSERVABILITY_SPEC.md\n• Understand architecture and integration points\n• Validate technical approach\n• Confirm timeline and resource allocation",
                        "priority": "high"
                    },
                    {
                        "name": "🏢 Create Sentry Organization & Projects",
                        "notes": "Set up Sentry infrastructure\n• Create organization account\n• Set up projects for each service (6 microservices)\n• Generate DSN keys for dev/staging/prod environments\n• Configure team access and permissions",
                        "priority": "high"
                    },
                    {
                        "name": "🔐 Configure Environment Variables & Secrets",
                        "notes": "Secure configuration management\n• Add SENTRY_DSN_* to environment configs\n• Set up feature flags for gradual rollout\n• Configure sampling rates per environment\n• Validate security and compliance requirements",
                        "priority": "medium"
                    }
                ]
            },
            
            # Phase 1: Foundation Setup
            {
                "section": "🏗️ Phase 1: Foundation Setup (Week 1-2)",
                "due_offset_days": 14,
                "tasks": [
                    {
                        "name": "📦 Install Sentry SDK Dependencies",
                        "notes": "Add sentry-sdk[fastapi,celery]>=1.40.0 to pyproject.toml\n• Update all service requirements\n• Test dependency compatibility\n• Run uv sync to install packages",
                        "priority": "high"
                    },
                    {
                        "name": "⚙️ Implement Core Sentry Configuration",
                        "notes": "Create common/observability/sentry_config.py\n• Environment-specific configuration\n• Trading-specific context management\n• Error filtering and classification\n• Integration with existing logging",
                        "priority": "high"
                    },
                    {
                        "name": "🔌 Integrate FastAPI Services (6 services)",
                        "notes": "Add Sentry to all FastAPI microservices:\n• Calendar API (9001)\n• Equity API (9002) \n• Option History API (9003)\n• Option Chain API (9004)\n• Orchestrator API (9005)\n• IB API Service (9010)",
                        "priority": "high",
                        "subtasks": [
                            "Calendar API Integration",
                            "Equity API Integration", 
                            "Option History API Integration",
                            "Option Chain API Integration",
                            "Orchestrator API Integration",
                            "IB API Service Integration"
                        ]
                    },
                    {
                        "name": "🏷️ Implement Trading Context Management",
                        "notes": "Enhance errors with trading-specific context\n• Symbol, market session, data source tagging\n• Correlation ID preservation\n• Business context (market hours, service health)\n• Error classification (market data, rate limiting, data integrity)",
                        "priority": "medium"
                    },
                    {
                        "name": "🛡️ Add Enhanced Error Handling Middleware",
                        "notes": "Create TradingObservabilityMiddleware\n• Extract trading context from requests\n• Set Sentry context with correlation IDs\n• Integrate with existing logging context\n• Test error capture and context propagation",
                        "priority": "medium"
                    },
                    {
                        "name": "✅ Phase 1 Testing & Validation",
                        "notes": "Validate core error tracking functionality\n• Test error capture across all services\n• Verify context and correlation ID propagation\n• Check Sentry dashboard data\n• Performance impact assessment",
                        "priority": "high"
                    }
                ]
            },
            
            # Phase 2: Distributed Tracing
            {
                "section": "🔗 Phase 2: Distributed Tracing (Week 3-4)",
                "due_offset_days": 28,
                "tasks": [
                    {
                        "name": "🌐 Implement Cross-Service HTTP Tracing",
                        "notes": "Enhance common/utils/http_client.py\n• Add trace header propagation\n• Track HTTP client performance\n• Monitor external API calls (IB API)\n• Test trace continuity across services",
                        "priority": "high"
                    },
                    {
                        "name": "🗄️ Add Database Query Tracing",
                        "notes": "Enhance repository layer with tracing\n• Track Parquet file operations\n• Monitor query performance\n• Add data volume metrics\n• Test repository operation visibility",
                        "priority": "high"
                    },
                    {
                        "name": "⚡ Implement Celery Task Distributed Tracing",
                        "notes": "Create TracedTradingTask base class\n• Task chain visibility\n• Queue performance monitoring\n• Correlation ID propagation through task chains\n• Rate limiting compliance tracking",
                        "priority": "high"
                    },
                    {
                        "name": "🔄 Add Workflow Tracing in Orchestrator",
                        "notes": "Enhance services/orchestrator_api/workflows.py\n• End-to-end workflow visibility\n• Step-by-step operation tracking\n• Data pipeline monitoring\n• Historical data fetch tracing",
                        "priority": "medium"
                    },
                    {
                        "name": "📊 Establish Performance Baselines",
                        "notes": "Collect baseline metrics for optimization\n• API response times\n• Database query performance\n• Celery task execution times\n• Memory and CPU utilization",
                        "priority": "medium"
                    },
                    {
                        "name": "✅ Phase 2 Testing & Validation",
                        "notes": "Validate distributed tracing functionality\n• Test end-to-end trace visibility\n• Verify performance data collection\n• Check correlation across services\n• Load test trace overhead",
                        "priority": "high"
                    }
                ]
            },
            
            # Phase 3: Advanced Features
            {
                "section": "📊 Phase 3: Advanced Features (Week 5-6)",
                "due_offset_days": 42,
                "tasks": [
                    {
                        "name": "📈 Implement Custom Trading Metrics",
                        "notes": "Create common/observability/metrics.py\n• Data freshness tracking\n• Market data quality monitoring\n• API rate limiting compliance\n• Business KPI measurement",
                        "priority": "high"
                    },
                    {
                        "name": "🖥️ Add Frontend Error Tracking (Next.js)",
                        "notes": "Integrate Sentry into frontend-nextjs/\n• Browser error tracking\n• Trading UI error context\n• API connection monitoring\n• User experience metrics",
                        "priority": "medium"
                    },
                    {
                        "name": "🚨 Configure Alerting & Notifications",
                        "notes": "Set up business-critical alert rules\n• Market data outage alerts\n• IB API disconnection monitoring\n• High error rate notifications\n• SLA breach alerting",
                        "priority": "high"
                    },
                    {
                        "name": "📊 Create Custom Dashboards",
                        "notes": "Build trading-specific Sentry dashboards\n• Service health overview\n• Trading performance metrics\n• Data pipeline status\n• Error trend analysis",
                        "priority": "medium"
                    },
                    {
                        "name": "🔍 Implement Business Intelligence Features",
                        "notes": "Advanced monitoring for trading operations\n• Symbol-specific performance tracking\n• Market hours vs off-hours analysis\n• Data source reliability metrics\n• Compliance and audit trail features",
                        "priority": "low"
                    },
                    {
                        "name": "✅ Phase 3 Testing & Validation",
                        "notes": "Validate advanced monitoring features\n• Test custom metrics collection\n• Verify alerting functionality\n• Check dashboard accuracy\n• Frontend integration testing",
                        "priority": "high"
                    }
                ]
            },
            
            # Phase 4: Testing & Optimization
            {
                "section": "🧪 Phase 4: Testing & Optimization (Week 7)",
                "due_offset_days": 49,
                "tasks": [
                    {
                        "name": "🚀 Conduct Load Testing with Observability",
                        "notes": "Test system performance with full observability\n• Measure Sentry integration overhead\n• Validate under high load conditions\n• Test alert responsiveness\n• Performance optimization",
                        "priority": "high"
                    },
                    {
                        "name": "🔧 Alert Tuning & False Positive Reduction",
                        "notes": "Optimize alerting system\n• Reduce alert noise\n• Tune severity levels\n• Adjust notification channels\n• Test escalation procedures",
                        "priority": "medium"
                    },
                    {
                        "name": "⚡ Performance Optimization",
                        "notes": "Optimize observability performance impact\n• Adjust sampling rates\n• Optimize trace data\n• Memory usage optimization\n• Response time validation",
                        "priority": "medium"
                    },
                    {
                        "name": "🎯 Final Integration Testing",
                        "notes": "End-to-end system validation\n• All services integration test\n• Cross-service tracing validation\n• Alert and notification testing\n• Rollback procedure testing",
                        "priority": "high"
                    },
                    {
                        "name": "🚀 Production Deployment Preparation",
                        "notes": "Prepare for production rollout\n• Environment configuration validation\n• Security review completion\n• Rollback procedures documented\n• Go-live checklist creation",
                        "priority": "high"
                    }
                ]
            },
            
            # Documentation & Training
            {
                "section": "📚 Documentation & Training",
                "due_offset_days": 56,
                "tasks": [
                    {
                        "name": "📖 Create Operational Runbooks",
                        "notes": "Documentation for ongoing operations\n• Incident response procedures\n• Alert investigation guides\n• Performance troubleshooting\n• Maintenance procedures",
                        "priority": "medium"
                    },
                    {
                        "name": "🎓 Team Training & Knowledge Transfer",
                        "notes": "Prepare team for observability operations\n• Sentry platform training\n• Custom metrics explanation\n• Alert handling procedures\n• Dashboard usage training",
                        "priority": "medium"
                    },
                    {
                        "name": "📋 Update Deployment Documentation",
                        "notes": "Update existing documentation\n• Add observability configuration to deployment guides\n• Update troubleshooting documentation\n• Environment setup instructions\n• Configuration management guides",
                        "priority": "low"
                    },
                    {
                        "name": "✅ Project Completion & Handover",
                        "notes": "Finalize project delivery\n• Complete all deliverables checklist\n• Conduct final review meeting\n• Document lessons learned\n• Plan future enhancements",
                        "priority": "high"
                    }
                ]
            }
        ]
        
        for group in task_groups:
            section_gid = sections.get(group["section"])
            if not section_gid:
                continue
            
            # Calculate due date for this phase
            due_date = None
            if "due_offset_days" in group:
                due_date = (start_date + timedelta(days=group["due_offset_days"])).strftime("%Y-%m-%d")
            
            print(f"\n   Creating tasks for: {group['section']}")
            
            for task_data in group["tasks"]:
                try:
                    # Create main task
                    task_request = {
                        "data": {
                            "name": task_data["name"],
                            "notes": task_data["notes"],
                            "projects": [project_gid],
                            "memberships": [{
                                "project": project_gid,
                                "section": section_gid
                            }]
                        }
                    }
                    
                    # Add due date if specified
                    if due_date:
                        task_request["data"]["due_on"] = due_date
                    
                    # Add priority if specified
                    if "priority" in task_data:
                        priority_map = {"high": "high", "medium": "medium", "low": "low"}
                        if task_data["priority"] in priority_map:
                            task_request["data"]["priority"] = priority_map[task_data["priority"]]
                    
                    task = self.tasks_api.create_task(task_request, {})
                    print(f"     ✅ {task_data['name']}")
                    
                    # Create subtasks if specified
                    if "subtasks" in task_data:
                        for subtask_name in task_data["subtasks"]:
                            subtask_request = {
                                "data": {
                                    "name": subtask_name,
                                    "parent": task["gid"],
                                    "projects": [project_gid]
                                }
                            }
                            
                            if due_date:
                                subtask_request["data"]["due_on"] = due_date
                            
                            self.tasks_api.create_task(subtask_request, {})
                            print(f"       • {subtask_name}")
                    
                except ApiException as e:
                    print(f"     ❌ Error creating task {task_data['name']}: {e}")
    
    def run(self):
        """Execute the full project creation"""
        print("🚀 Creating Asana project for Trading Platform Observability...")
        print(f"   Workspace: {self.workspace_gid}")
        
        # Create project
        project_gid = self.create_project()
        
        # Create sections
        print("\n📋 Creating project sections...")
        sections = self.create_sections(project_gid)
        
        # Create tasks
        print("\n✅ Creating tasks...")
        self.create_tasks(project_gid, sections)
        
        print(f"\n🎉 Successfully created observability project!")
        print(f"   Total sections: {len(sections)}")
        print(f"   Project GID: {project_gid}")
        print(f"\n💡 Next steps:")
        print(f"   1. Review the created project in Asana")
        print(f"   2. Assign team members to relevant tasks")
        print(f"   3. Adjust due dates based on your timeline")
        print(f"   4. Begin Phase 1 implementation")


def main():
    parser = argparse.ArgumentParser(description="Create Asana project for Trading Platform Observability")
    parser.add_argument("--token", required=True, help="Asana personal access token")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be created without creating")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual project will be created")
        print("   Project: 🔍 Trading Platform - Observability & Sentry Integration")
        print("   Sections: 6 phases + documentation")
        print("   Tasks: ~30 tasks with subtasks and dependencies")
        return
    
    try:
        creator = AsanaObservabilityProject(args.token)
        creator.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()