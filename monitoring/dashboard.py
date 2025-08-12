#!/usr/bin/env python3
"""
Trading Data System Monitoring Dashboard

Real-time monitoring and alerting system for the comprehensive daily update
system with earnings prioritization.

Features:
- Real-time execution monitoring
- Performance metrics and trends
- Earnings calendar integration
- Alert system for failures and anomalies
- Historical performance analysis
- System health checks

Usage:
    # Start monitoring dashboard
    python monitoring/dashboard.py --port 8080
    
    # Generate static report
    python monitoring/dashboard.py --report --output reports/
"""

import json
import logging
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3

# Optional web dashboard dependencies
try:
    from flask import Flask, render_template, jsonify, request
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    WEB_DASHBOARD_AVAILABLE = True
except ImportError:
    WEB_DASHBOARD_AVAILABLE = False

# Core system imports
from earnings.fetcher import EarningsCalendarFetcher
from earnings.scheduler import EarningsPriorityScheduler, SchedulePriority
from jobs.orchestrator import DataPipelineOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Execution metrics for monitoring."""
    timestamp: datetime
    symbols_processed: int
    successful_symbols: int
    failed_symbols: int
    total_duration_seconds: float
    avg_symbol_duration_seconds: float
    earnings_events_processed: int
    priority_distribution: Dict[str, int]
    circuit_breakers_active: int
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


@dataclass
class AlertEvent:
    """Alert event for monitoring."""
    timestamp: datetime
    alert_type: str  # 'error', 'warning', 'info'
    severity: str   # 'critical', 'high', 'medium', 'low'
    message: str
    details: Optional[Dict[str, Any]] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MonitoringDatabase:
    """SQLite database for monitoring data persistence."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize monitoring database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbols_processed INTEGER,
                    successful_symbols INTEGER,
                    failed_symbols INTEGER,
                    total_duration_seconds REAL,
                    avg_symbol_duration_seconds REAL,
                    earnings_events_processed INTEGER,
                    priority_distribution TEXT,  -- JSON
                    circuit_breakers_active INTEGER,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,  -- JSON
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON execution_metrics(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                ON alert_events(timestamp)
            """)
    
    def save_metrics(self, metrics: ExecutionMetrics):
        """Save execution metrics to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO execution_metrics (
                    timestamp, symbols_processed, successful_symbols, failed_symbols,
                    total_duration_seconds, avg_symbol_duration_seconds,
                    earnings_events_processed, priority_distribution,
                    circuit_breakers_active, memory_usage_mb, cpu_usage_percent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.symbols_processed,
                metrics.successful_symbols,
                metrics.failed_symbols,
                metrics.total_duration_seconds,
                metrics.avg_symbol_duration_seconds,
                metrics.earnings_events_processed,
                json.dumps(metrics.priority_distribution),
                metrics.circuit_breakers_active,
                metrics.memory_usage_mb,
                metrics.cpu_usage_percent
            ))
    
    def save_alert(self, alert: AlertEvent):
        """Save alert event to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO alert_events (
                    timestamp, alert_type, severity, message, details, resolved, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.timestamp.isoformat(),
                alert.alert_type,
                alert.severity,
                alert.message,
                json.dumps(alert.details) if alert.details else None,
                1 if alert.resolved else 0,
                alert.resolved_at.isoformat() if alert.resolved_at else None
            ))
    
    def get_recent_metrics(self, hours: int = 24) -> List[ExecutionMetrics]:
        """Get recent execution metrics."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM execution_metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """, (cutoff.isoformat(),))
            
            metrics = []
            for row in cursor:
                metrics.append(ExecutionMetrics(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    symbols_processed=row['symbols_processed'],
                    successful_symbols=row['successful_symbols'],
                    failed_symbols=row['failed_symbols'],
                    total_duration_seconds=row['total_duration_seconds'],
                    avg_symbol_duration_seconds=row['avg_symbol_duration_seconds'],
                    earnings_events_processed=row['earnings_events_processed'],
                    priority_distribution=json.loads(row['priority_distribution'] or '{}'),
                    circuit_breakers_active=row['circuit_breakers_active'],
                    memory_usage_mb=row['memory_usage_mb'],
                    cpu_usage_percent=row['cpu_usage_percent']
                ))
            
            return metrics
    
    def get_recent_alerts(self, hours: int = 24) -> List[AlertEvent]:
        """Get recent alert events."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM alert_events 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """, (cutoff.isoformat(),))
            
            alerts = []
            for row in cursor:
                alerts.append(AlertEvent(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    alert_type=row['alert_type'],
                    severity=row['severity'],
                    message=row['message'],
                    details=json.loads(row['details']) if row['details'] else None,
                    resolved=bool(row['resolved']),
                    resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None
                ))
            
            return alerts


class SystemMonitor:
    """Real-time system monitoring and alerting."""
    
    def __init__(self, 
                 base_path: Path,
                 db_path: Path = None):
        
        self.base_path = base_path
        self.db = MonitoringDatabase(db_path or base_path / "monitoring" / "monitoring.db")
        
        # System components (for health checks)
        self.orchestrator = None
        self.scheduler = None
        self.earnings_fetcher = None
        
        # Real-time tracking
        self.current_execution = None
        self.recent_metrics = deque(maxlen=100)  # Keep last 100 metrics
        self.active_alerts = {}
        
        # Thresholds for alerts
        self.alert_thresholds = {
            'failure_rate_critical': 0.25,  # 25% failure rate
            'failure_rate_warning': 0.10,   # 10% failure rate
            'duration_warning_minutes': 60,  # 1 hour execution time
            'duration_critical_minutes': 120, # 2 hour execution time
            'circuit_breaker_warning': 3,    # 3 circuit breakers
            'circuit_breaker_critical': 5    # 5 circuit breakers
        }
    
    def initialize_components(self):
        """Initialize system components for monitoring."""
        try:
            self.orchestrator = DataPipelineOrchestrator(self.base_path, enable_lineage=True)
            self.earnings_fetcher = EarningsCalendarFetcher()
            self.scheduler = EarningsPriorityScheduler(
                base_path=self.base_path,
                orchestrator=self.orchestrator,
                earnings_fetcher=self.earnings_fetcher
            )
            logger.info("✅ System components initialized for monitoring")
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            # Get system resource usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
        except ImportError:
            memory_mb = None
            cpu_percent = None
        
        # Get scheduler status if available
        scheduler_status = {}
        if self.scheduler:
            scheduler_status = self.scheduler.get_status_report()
        
        # Get data directory stats
        data_stats = self._get_data_directory_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_resources': {
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
            },
            'scheduler_status': scheduler_status,
            'data_stats': data_stats,
            'components_initialized': {
                'orchestrator': self.orchestrator is not None,
                'scheduler': self.scheduler is not None,
                'earnings_fetcher': self.earnings_fetcher is not None
            }
        }
    
    def _get_data_directory_stats(self) -> Dict[str, Any]:
        """Get data directory statistics."""
        try:
            total_size = 0
            file_counts = defaultdict(int)
            
            for path in self.base_path.rglob("*.parquet"):
                total_size += path.stat().st_size
                file_counts[path.parent.name] += 1
            
            return {
                'total_size_mb': total_size / (1024 * 1024),
                'file_counts': dict(file_counts),
                'total_files': sum(file_counts.values())
            }
        except Exception as e:
            logger.error(f"Failed to get data stats: {e}")
            return {}
    
    def analyze_execution_results(self, execution_results: Dict[str, Any]) -> ExecutionMetrics:
        """Analyze execution results and generate metrics."""
        
        total_duration = execution_results.get('total_duration', 0)
        symbols_processed = execution_results.get('total_symbols_processed', 0)
        successful = execution_results.get('successful_symbols', 0)
        failed = execution_results.get('failed_symbols', 0)
        circuit_breakers = execution_results.get('circuit_breakers_triggered', 0)
        
        # Calculate average duration per symbol
        avg_duration = total_duration / max(symbols_processed, 1)
        
        # Get priority distribution from batches
        priority_dist = defaultdict(int)
        for batch in execution_results.get('batches', []):
            priority = batch.get('priority', 'unknown')
            symbols_in_batch = batch.get('symbols_processed', 0)
            priority_dist[priority] += symbols_in_batch
        
        # Get system resources
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
        except ImportError:
            memory_mb = None
            cpu_percent = None
        
        metrics = ExecutionMetrics(
            timestamp=datetime.now(),
            symbols_processed=symbols_processed,
            successful_symbols=successful,
            failed_symbols=failed,
            total_duration_seconds=total_duration,
            avg_symbol_duration_seconds=avg_duration,
            earnings_events_processed=0,  # Will be filled by caller if available
            priority_distribution=dict(priority_dist),
            circuit_breakers_active=circuit_breakers,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent
        )
        
        # Save metrics
        self.db.save_metrics(metrics)
        self.recent_metrics.append(metrics)
        
        # Check for alerts
        self._check_execution_alerts(metrics)
        
        return metrics
    
    def _check_execution_alerts(self, metrics: ExecutionMetrics):
        """Check execution metrics for alert conditions."""
        
        # Calculate failure rate
        total_symbols = metrics.symbols_processed
        if total_symbols > 0:
            failure_rate = metrics.failed_symbols / total_symbols
            
            # Failure rate alerts
            if failure_rate >= self.alert_thresholds['failure_rate_critical']:
                self._create_alert(
                    'error', 'critical',
                    f"Critical failure rate: {failure_rate:.1%} ({metrics.failed_symbols}/{total_symbols})",
                    {'failure_rate': failure_rate, 'failed_symbols': metrics.failed_symbols}
                )
            elif failure_rate >= self.alert_thresholds['failure_rate_warning']:
                self._create_alert(
                    'warning', 'high',
                    f"High failure rate: {failure_rate:.1%} ({metrics.failed_symbols}/{total_symbols})",
                    {'failure_rate': failure_rate, 'failed_symbols': metrics.failed_symbols}
                )
        
        # Duration alerts
        duration_minutes = metrics.total_duration_seconds / 60
        if duration_minutes >= self.alert_thresholds['duration_critical_minutes']:
            self._create_alert(
                'warning', 'high',
                f"Long execution time: {duration_minutes:.1f} minutes",
                {'duration_seconds': metrics.total_duration_seconds}
            )
        elif duration_minutes >= self.alert_thresholds['duration_warning_minutes']:
            self._create_alert(
                'warning', 'medium',
                f"Extended execution time: {duration_minutes:.1f} minutes",
                {'duration_seconds': metrics.total_duration_seconds}
            )
        
        # Circuit breaker alerts
        if metrics.circuit_breakers_active >= self.alert_thresholds['circuit_breaker_critical']:
            self._create_alert(
                'error', 'critical',
                f"Multiple circuit breakers active: {metrics.circuit_breakers_active}",
                {'circuit_breakers_active': metrics.circuit_breakers_active}
            )
        elif metrics.circuit_breakers_active >= self.alert_thresholds['circuit_breaker_warning']:
            self._create_alert(
                'warning', 'high',
                f"Circuit breakers active: {metrics.circuit_breakers_active}",
                {'circuit_breakers_active': metrics.circuit_breakers_active}
            )
    
    def _create_alert(self, alert_type: str, severity: str, message: str, details: Dict[str, Any] = None):
        """Create and save alert."""
        
        # Create alert key to avoid duplicates
        alert_key = f"{alert_type}_{severity}_{hash(message)}"
        
        # Don't create duplicate alerts within 1 hour
        if alert_key in self.active_alerts:
            last_alert_time = self.active_alerts[alert_key]
            if (datetime.now() - last_alert_time).total_seconds() < 3600:
                return
        
        alert = AlertEvent(
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details
        )
        
        self.db.save_alert(alert)
        self.active_alerts[alert_key] = alert.timestamp
        
        # Log alert
        log_level = {
            'critical': logging.CRITICAL,
            'high': logging.ERROR,
            'medium': logging.WARNING,
            'low': logging.INFO
        }.get(severity, logging.INFO)
        
        logger.log(log_level, f"🚨 ALERT [{severity.upper()}]: {message}")
    
    def generate_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        # Get recent metrics
        recent_metrics = self.db.get_recent_metrics(hours=days * 24)
        recent_alerts = self.db.get_recent_alerts(hours=days * 24)
        
        if not recent_metrics:
            return {'error': 'No recent metrics available'}
        
        # Calculate performance statistics
        total_executions = len(recent_metrics)
        total_symbols = sum(m.symbols_processed for m in recent_metrics)
        total_successful = sum(m.successful_symbols for m in recent_metrics)
        total_failed = sum(m.failed_symbols for m in recent_metrics)
        
        avg_duration = sum(m.total_duration_seconds for m in recent_metrics) / total_executions
        avg_success_rate = total_successful / max(total_symbols, 1)
        
        # Alert statistics
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[f"{alert.alert_type}_{alert.severity}"] += 1
        
        # Priority distribution over time
        priority_trends = defaultdict(list)
        for metrics in recent_metrics:
            for priority, count in metrics.priority_distribution.items():
                priority_trends[priority].append(count)
        
        return {
            'report_period_days': days,
            'report_generated': datetime.now().isoformat(),
            'execution_summary': {
                'total_executions': total_executions,
                'total_symbols_processed': total_symbols,
                'total_successful': total_successful,
                'total_failed': total_failed,
                'overall_success_rate': avg_success_rate,
                'average_execution_duration_minutes': avg_duration / 60
            },
            'alert_summary': dict(alert_counts),
            'priority_trends': dict(priority_trends),
            'recent_metrics': [asdict(m) for m in recent_metrics[-10:]],  # Last 10
            'recent_alerts': [asdict(a) for a in recent_alerts[-20:]]     # Last 20
        }


# Web Dashboard (optional)
if WEB_DASHBOARD_AVAILABLE:
    
    def create_web_dashboard(monitor: SystemMonitor, port: int = 8080):
        """Create Flask web dashboard for monitoring."""
        
        app = Flask(__name__)
        app.json_encoder = PlotlyJSONEncoder
        
        @app.route('/')
        def dashboard():
            """Main dashboard page."""
            return render_template('dashboard.html')
        
        @app.route('/api/status')
        def api_status():
            """Current system status API."""
            return jsonify(monitor.collect_system_metrics())
        
        @app.route('/api/metrics')
        def api_metrics():
            """Recent metrics API."""
            hours = request.args.get('hours', 24, type=int)
            metrics = monitor.db.get_recent_metrics(hours)
            return jsonify([asdict(m) for m in metrics])
        
        @app.route('/api/alerts')
        def api_alerts():
            """Recent alerts API."""
            hours = request.args.get('hours', 24, type=int)
            alerts = monitor.db.get_recent_alerts(hours)
            return jsonify([asdict(a) for a in alerts])
        
        @app.route('/api/report')
        def api_report():
            """Performance report API."""
            days = request.args.get('days', 7, type=int)
            return jsonify(monitor.generate_performance_report(days))
        
        @app.route('/api/charts/performance')
        def api_performance_chart():
            """Performance metrics chart."""
            metrics = monitor.db.get_recent_metrics(24)
            
            if not metrics:
                return jsonify({'error': 'No data available'})
            
            # Create performance chart
            timestamps = [m.timestamp for m in metrics]
            success_rates = [m.successful_symbols / max(m.symbols_processed, 1) for m in metrics]
            durations = [m.total_duration_seconds / 60 for m in metrics]  # Minutes
            
            fig = go.Figure()
            
            # Success rate line
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=success_rates,
                mode='lines+markers',
                name='Success Rate',
                yaxis='y'
            ))
            
            # Duration line (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=durations,
                mode='lines+markers',
                name='Duration (min)',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='System Performance Over Time',
                xaxis_title='Time',
                yaxis=dict(title='Success Rate', side='left'),
                yaxis2=dict(title='Duration (minutes)', side='right', overlaying='y')
            )
            
            return json.dumps(fig, cls=PlotlyJSONEncoder)
        
        logger.info(f"🌐 Starting web dashboard on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)


def main():
    """Main monitoring entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading system monitoring dashboard")
    
    parser.add_argument("--data-path", type=Path, default=Path("data"),
                       help="Base data path")
    parser.add_argument("--db-path", type=Path,
                       help="Monitoring database path")
    parser.add_argument("--port", type=int, default=8080,
                       help="Web dashboard port")
    parser.add_argument("--report", action="store_true",
                       help="Generate static report instead of web dashboard")
    parser.add_argument("--output", type=Path,
                       help="Output directory for static report")
    parser.add_argument("--days", type=int, default=7,
                       help="Report period in days")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = SystemMonitor(args.data_path, args.db_path)
    monitor.initialize_components()
    
    if args.report:
        # Generate static report
        print("📊 Generating monitoring report...")
        
        report = monitor.generate_performance_report(args.days)
        
        output_dir = args.output or Path("reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        report_file = output_dir / f"monitoring_report_{date.today().isoformat()}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report saved to: {report_file}")
        
        # Print summary
        if 'execution_summary' in report:
            summary = report['execution_summary']
            print(f"\n📈 Performance Summary ({args.days} days):")
            print(f"  Executions: {summary['total_executions']}")
            print(f"  Symbols processed: {summary['total_symbols_processed']:,}")
            print(f"  Success rate: {summary['overall_success_rate']:.1%}")
            print(f"  Avg duration: {summary['average_execution_duration_minutes']:.1f} minutes")
    
    elif WEB_DASHBOARD_AVAILABLE:
        # Start web dashboard
        create_web_dashboard(monitor, args.port)
    
    else:
        print("❌ Web dashboard dependencies not available")
        print("Install with: pip install flask plotly")
        print("Generating static report instead...")
        
        report = monitor.generate_performance_report(args.days)
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()