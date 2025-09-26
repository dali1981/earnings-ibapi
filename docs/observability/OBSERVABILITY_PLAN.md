# Observability Implementation Plan - Trading Platform
*Detailed execution roadmap for Sentry integration and enhanced monitoring*

## Executive Summary

This plan provides a step-by-step implementation guide for integrating comprehensive observability into our trading platform using Sentry. The implementation is structured in 4 phases over 7 weeks, building incrementally on our existing robust logging and monitoring infrastructure.

**Timeline**: 7 weeks total
**Resources**: 1-2 developers
**Budget**: ~$80/month for Sentry Organization Plan
**Risk Level**: Low (gradual rollout with rollback capabilities)

---

## Phase 1: Foundation Setup (Weeks 1-2)

### Week 1: Core Infrastructure

#### Day 1-2: Environment Setup & Project Creation

**Tasks:**
1. **Sentry Project Setup**
   ```bash
   # Create Sentry organization account
   # Set up projects for each service:
   - trading-platform-calendar-api
   - trading-platform-equity-api  
   - trading-platform-option-history-api
   - trading-platform-option-chain-api
   - trading-platform-orchestrator-api
   - trading-platform-ib-api
   - trading-platform-frontend
   ```

2. **Environment Configuration**
   ```python
   # Create common/observability/config.py
   SENTRY_ENVIRONMENTS = {
       'development': {
           'dsn': 'https://xxx@xxx.ingest.sentry.io/development',
           'traces_sample_rate': 1.0,
           'profiles_sample_rate': 1.0,
           'debug': True
       },
       'production': {
           'dsn': 'https://xxx@xxx.ingest.sentry.io/production', 
           'traces_sample_rate': 0.1,
           'profiles_sample_rate': 0.05,
           'debug': False
       }
   }
   ```

3. **Dependencies Installation**
   ```bash
   # Add to pyproject.toml
   uv add "sentry-sdk[fastapi,celery]>=1.40.0"
   ```

**Deliverables:**
- [ ] Sentry organization and projects created
- [ ] Environment-specific DSN keys generated  
- [ ] Base configuration module created
- [ ] Dependencies added to project

#### Day 3-5: Core SDK Integration

**Tasks:**
1. **Create Base Sentry Configuration**
   ```python
   # File: common/observability/sentry_config.py
   import sentry_sdk
   from sentry_sdk.integrations.fastapi import FastApiIntegration
   from sentry_sdk.integrations.celery import CeleryIntegration
   from sentry_sdk.integrations.logging import LoggingIntegration
   from app_settings.settings import get_settings

   def configure_sentry(service_name: str):
       settings = get_settings()
       
       if not settings.SENTRY_DSN:
           return  # Skip if no DSN configured
           
       sentry_sdk.init(
           dsn=settings.SENTRY_DSN,
           environment=settings.APP_ENV,
           traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
           profiles_sample_rate=settings.SENTRY_PROFILES_SAMPLE_RATE,
           send_default_pii=False,
           
           server_name=service_name,
           release=get_app_version(),
           
           integrations=[
               FastApiIntegration(
                   auto_enabling_integrations=True,
                   transaction_style="endpoint"
               ),
               CeleryIntegration(
                   propagate_traces=True,
                   monitor_beat_tasks=True
               ),
               LoggingIntegration(
                   level=logging.INFO,
                   event_level=logging.WARNING
               )
           ],
           
           before_send=trading_error_filter,
           before_send_transaction=trading_transaction_filter
       )
   ```

2. **Add Configuration to Settings**
   ```python
   # File: app_settings/settings.py (additions)
   class Settings(BaseSettings):
       # ... existing settings ...
       
       # Sentry Configuration
       SENTRY_DSN: Optional[str] = None
       SENTRY_TRACES_SAMPLE_RATE: float = 0.1
       SENTRY_PROFILES_SAMPLE_RATE: float = 0.05
       OBSERVABILITY_ENABLED: bool = True
   ```

3. **Integrate with Each Service**
   ```python
   # File: services/calendar_api/main.py
   from common.observability.sentry_config import configure_sentry
   
   # Add after FastAPI app creation, before routes
   configure_sentry("calendar-api")
   ```

**Deliverables:**
- [ ] Base Sentry configuration module
- [ ] Settings integration for DSN and sampling rates
- [ ] All 6 microservices integrated with basic Sentry
- [ ] Basic error tracking operational

### Week 2: Error Context & Classification

#### Day 6-8: Trading-Specific Error Context

**Tasks:**
1. **Create Trading Context Manager**
   ```python
   # File: common/observability/trading_context.py
   import sentry_sdk
   from datetime import datetime
   from typing import Optional
   from common.utils.market_hours import is_market_hours
   
   class TradingContextManager:
       @staticmethod
       def set_trading_context(
           symbol: Optional[str] = None,
           market_session: Optional[str] = None,
           data_source: Optional[str] = None,
           correlation_id: Optional[str] = None,
           operation: Optional[str] = None
       ):
           with sentry_sdk.configure_scope() as scope:
               # Trading tags
               if symbol:
                   scope.set_tag("trading.symbol", symbol)
               if market_session:
                   scope.set_tag("trading.session", market_session)  
               if data_source:
                   scope.set_tag("trading.data_source", data_source)
               if operation:
                   scope.set_tag("trading.operation", operation)
                   
               # Correlation tracking
               if correlation_id:
                   scope.set_context("correlation", {"id": correlation_id})
               
               # Business context
               scope.set_context("trading", {
                   "timestamp": datetime.utcnow().isoformat(),
                   "market_hours": is_market_hours(),
                   "service_health": "healthy"  # TODO: integrate with health checks
               })
   ```

2. **Create Error Classification System**
   ```python
   # File: common/observability/error_classification.py
   import sentry_sdk
   from typing import Dict, Any, Optional
   
   def trading_error_filter(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
       """Filter and enhance trading-specific errors"""
       
       if not event.get('exception'):
           return event
           
       error_type = event['exception']['values'][0]['type']
       error_value = event['exception']['values'][0]['value']
       
       # Market data errors - business critical
       if 'market_data' in error_value.lower() or 'ib_api' in error_value.lower():
           event['level'] = 'fatal'
           event['fingerprint'] = ['market-data', error_type]
           
       # Rate limiting - expected during high load
       elif 'rate limit' in error_value.lower():
           event['level'] = 'warning'
           event['fingerprint'] = ['rate-limit', event.get('transaction')]
           
       # Data integrity issues
       elif any(term in error_value.lower() for term in ['duplicate', 'missing_data', 'invalid_data']):
           event['level'] = 'error'
           event['fingerprint'] = ['data-integrity', event.get('contexts', {}).get('trading', {}).get('symbol')]
           
       # Connection issues
       elif any(term in error_value.lower() for term in ['connection', 'timeout', 'network']):
           event['level'] = 'warning'
           event['fingerprint'] = ['connectivity', error_type]
           
       return event
   
   def trading_transaction_filter(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
       """Filter performance transactions"""
       
       # Skip health check transactions
       if event.get('transaction') == '/health':
           return None
           
       # Sample long-running operations differently
       if event.get('contexts', {}).get('trace', {}).get('op') == 'celery.task':
           # Always capture Celery task traces
           return event
           
       return event
   ```

**Deliverables:**
- [ ] Trading context manager for business-specific tagging
- [ ] Error classification system with trading-specific logic
- [ ] Enhanced error fingerprinting for better grouping

#### Day 9-10: FastAPI Middleware Integration

**Tasks:**
1. **Create Observability Middleware**
   ```python
   # File: common/observability/middleware.py
   from fastapi import Request
   from starlette.middleware.base import BaseHTTPMiddleware
   from common.observability.trading_context import TradingContextManager
   from common.utils.correlation import generate_correlation_id
   from common.utils.logging_config import trading_context
   
   class TradingObservabilityMiddleware(BaseHTTPMiddleware):
       async def dispatch(self, request: Request, call_next):
           # Extract context from request
           symbol = request.path_params.get('symbol')
           correlation_id = request.headers.get('X-Correlation-ID') or generate_correlation_id()
           
           # Determine operation from path
           operation = self._extract_operation(request.url.path)
           
           # Set Sentry context
           TradingContextManager.set_trading_context(
               symbol=symbol,
               correlation_id=correlation_id,
               data_source="api_request",
               operation=operation
           )
           
           # Maintain existing logging context
           with trading_context(correlation_id=correlation_id, symbol=symbol):
               response = await call_next(request)
               
           return response
       
       def _extract_operation(self, path: str) -> str:
           """Extract operation type from URL path"""
           if '/bars' in path:
               return 'data_retrieval'
           elif '/dates' in path:
               return 'metadata_query'
           elif '/historical' in path:
               return 'data_ingestion'
           elif '/chains' in path:
               return 'option_chains'
           else:
               return 'general_api'
   ```

2. **Add Middleware to Services**
   ```python
   # File: services/equity_api/main.py (example)
   from common.observability.middleware import TradingObservabilityMiddleware
   
   app = FastAPI(title="Equity API", version="1.0.0")
   
   # Add observability middleware
   app.add_middleware(TradingObservabilityMiddleware)
   ```

**Deliverables:**
- [ ] Observability middleware created
- [ ] Middleware integrated across all services
- [ ] Request-level context tracking operational

---

## Phase 2: Distributed Tracing (Weeks 3-4)

### Week 3: Cross-Service Tracing

#### Day 11-13: HTTP Client Integration

**Tasks:**
1. **Create Traced HTTP Client**
   ```python
   # File: common/utils/traced_http_client.py
   import httpx
   import sentry_sdk
   from typing import Dict, Any, Optional
   
   class TracedHttpClient:
       def __init__(self):
           self.client = httpx.AsyncClient()
       
       async def request(self, method: str, url: str, **kwargs):
           with sentry_sdk.start_transaction(
               op="http.client",
               name=f"{method.upper()} {self._clean_url(url)}"
           ) as transaction:
               
               # Add trace headers
               headers = kwargs.get('headers', {})
               headers.update(sentry_sdk.Hub.current.trace_propagation_meta)
               kwargs['headers'] = headers
               
               try:
                   response = await self.client.request(method, url, **kwargs)
                   
                   # Add response metadata
                   transaction.set_tag("http.status_code", response.status_code)
                   transaction.set_data("response_size", len(response.content))
                   transaction.set_tag("http.url", self._clean_url(url))
                   
                   if response.status_code >= 400:
                       transaction.set_tag("error", True)
                   
                   return response
                   
               except Exception as e:
                   transaction.set_tag("error", True)
                   sentry_sdk.capture_exception(e)
                   raise
       
       def _clean_url(self, url: str) -> str:
           """Remove sensitive parameters from URL for logging"""
           # Remove query parameters that might contain sensitive data
           if '?' in url:
               return url.split('?')[0]
           return url
   ```

2. **Update Existing HTTP Usage**
   ```python
   # File: services/orchestrator_api/clients/service_client.py
   from common.utils.traced_http_client import TracedHttpClient
   
   class ServiceClient:
       def __init__(self):
           self.http_client = TracedHttpClient()
       
       async def get_equity_data(self, symbol: str, start: str, end: str):
           url = f"http://equity-api:9002/equity/{symbol}/bars"
           params = {"start": start, "end": end}
           
           response = await self.http_client.request(
               "GET", url, params=params
           )
           return response.json()
   ```

**Deliverables:**
- [ ] Traced HTTP client with automatic trace propagation
- [ ] Updated service clients to use traced HTTP client
- [ ] Cross-service trace correlation working

#### Day 14-15: Database Operation Tracing

**Tasks:**
1. **Enhanced Repository Tracing**
   ```python
   # File: common/repositories/traced_base.py
   import sentry_sdk
   import pandas as pd
   from typing import Dict, Any
   from common.repositories.base import BaseRepository
   
   class TracedBaseRepository(BaseRepository):
       def save(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
           with sentry_sdk.start_span(
               op="db.operation",
               description=f"Save to {self.__class__.__name__}"
           ) as span:
               # Add operation metadata
               span.set_data("rows_count", len(df))
               span.set_data("columns", list(df.columns))
               span.set_data("memory_usage_mb", df.memory_usage(deep=True).sum() / 1024 / 1024)
               span.set_tag("repository.type", getattr(self, 'repository_type', 'unknown'))
               span.set_tag("operation.type", "save")
               
               # Add business context
               if 'symbol' in kwargs:
                   span.set_tag("trading.symbol", kwargs['symbol'])
               
               try:
                   result = super().save(df, **kwargs)
                   
                   # Add result metadata
                   span.set_data("files_written", len(result.get('files', [])))
                   span.set_data("bytes_written", result.get('bytes_written', 0))
                   span.set_tag("operation.status", "success")
                   
                   return result
                   
               except Exception as e:
                   span.set_tag("operation.status", "error")
                   span.set_data("error_message", str(e))
                   raise
       
       def load(self, **kwargs) -> pd.DataFrame:
           with sentry_sdk.start_span(
               op="db.query", 
               description=f"Load from {self.__class__.__name__}"
           ) as span:
               # Add query parameters
               span.set_data("query_params", kwargs)
               span.set_tag("operation.type", "load")
               
               if 'symbol' in kwargs:
                   span.set_tag("trading.symbol", kwargs['symbol'])
               
               try:
                   df = super().load(**kwargs)
                   
                   # Add result metadata
                   span.set_data("rows_loaded", len(df))
                   span.set_data("columns_loaded", list(df.columns))
                   span.set_tag("operation.status", "success")
                   
                   return df
                   
               except Exception as e:
                   span.set_tag("operation.status", "error") 
                   span.set_data("error_message", str(e))
                   raise
   ```

2. **Update Repository Inheritance**
   ```python
   # File: common/repositories/equity_bars.py
   from common.repositories.traced_base import TracedBaseRepository
   
   class EquityBarRepository(TracedBaseRepository):
       # Existing implementation - just change base class
       pass
   ```

**Deliverables:**
- [ ] Enhanced repository base class with tracing
- [ ] All repositories updated to use traced base
- [ ] Database operation visibility in Sentry

### Week 4: Celery Task Tracing

#### Day 16-18: Task Instrumentation

**Tasks:**
1. **Create Traced Task Base Class**
   ```python
   # File: ops/tasks/traced_base.py
   import sentry_sdk
   from celery import Task
   from typing import Any, Dict
   from common.observability.trading_context import TradingContextManager
   
   class TracedTradingTask(Task):
       def __call__(self, *args, **kwargs):
           # Extract business context
           symbol = self._extract_symbol(*args, **kwargs)
           task_name = self.name.split('.')[-1]  # Get task name without module
           
           with sentry_sdk.start_transaction(
               op="celery.task",
               name=task_name,
               sampled=True
           ) as transaction:
               
               # Set trading context
               TradingContextManager.set_trading_context(
                   symbol=symbol,
                   data_source="celery_task",
                   correlation_id=self.request.id,
                   operation=task_name
               )
               
               # Add Celery metadata
               transaction.set_tag("celery.task_name", self.name)
               transaction.set_tag("celery.queue", self.request.delivery_info.get('routing_key'))
               transaction.set_tag("celery.retry_count", self.request.retries)
               transaction.set_data("celery.args", args)
               transaction.set_data("celery.kwargs", {k: v for k, v in kwargs.items() if not k.startswith('_')})
               
               try:
                   result = super().__call__(*args, **kwargs)
                   transaction.set_tag("task.status", "success")
                   
                   # Add result metadata if available
                   if isinstance(result, dict):
                       for key in ['processed', 'saved', 'skipped']:
                           if key in result:
                               transaction.set_data(f"result.{key}", result[key])
                   
                   return result
                   
               except Exception as exc:
                   transaction.set_tag("task.status", "failure")
                   transaction.set_data("error_type", exc.__class__.__name__)
                   sentry_sdk.capture_exception(exc)
                   raise
       
       def _extract_symbol(self, *args, **kwargs) -> str:
           """Extract symbol from task arguments"""
           # Check kwargs first
           if 'symbol' in kwargs:
               return kwargs['symbol']
           if 'symbols' in kwargs and kwargs['symbols']:
               return kwargs['symbols'][0] if isinstance(kwargs['symbols'], list) else str(kwargs['symbols'])
           
           # Check args
           if args:
               first_arg = args[0]
               if isinstance(first_arg, str) and len(first_arg) <= 10:  # Likely a symbol
                   return first_arg
               elif isinstance(first_arg, dict) and 'symbol' in first_arg:
                   return first_arg['symbol']
           
           return "unknown"
   ```

2. **Update Existing Tasks**
   ```python
   # File: ops/tasks/historical_data.py
   from ops.tasks.traced_base import TracedTradingTask
   from celery import shared_task
   
   @shared_task(base=TracedTradingTask, bind=True, queue='historical_data')
   def fetch_historical_bars(self, symbol: str, **params):
       # Existing implementation - base class handles tracing
       pass
   ```

**Deliverables:**
- [ ] Traced task base class with business context
- [ ] All Celery tasks updated to use traced base
- [ ] Task execution tracing operational

#### Day 19-20: Workflow Orchestration Tracing

**Tasks:**
1. **Enhanced Orchestrator Tracing**
   ```python
   # File: services/orchestrator_api/traced_workflows.py
   import sentry_sdk
   from typing import List, Dict, Any
   from services.orchestrator_api.workflows.base import BaseWorkflow
   
   class TracedOrchestrator(BaseWorkflow):
       async def execute_historical_data_pipeline(self, symbols: List[str], **params) -> Dict[str, Any]:
           with sentry_sdk.start_transaction(
               op="workflow.data_pipeline",
               name="Historical Data Pipeline"
           ) as transaction:
               
               transaction.set_data("symbols_count", len(symbols))
               transaction.set_data("symbols", symbols[:10])  # First 10 for visibility
               transaction.set_tag("workflow.type", "historical_data")
               transaction.set_data("pipeline_params", params)
               
               results = []
               
               for i, symbol in enumerate(symbols):
                   with sentry_sdk.start_span(
                       op="workflow.step",
                       description=f"Process {symbol} ({i+1}/{len(symbols)})"
                   ) as span:
                       
                       span.set_tag("trading.symbol", symbol)
                       span.set_data("step_index", i)
                       
                       try:
                           # Step 1: Contract resolution
                           with sentry_sdk.start_span(op="contract.resolve") as contract_span:
                               contract = await self.resolve_contract(symbol)
                               contract_span.set_data("contract_resolved", bool(contract))
                           
                           # Step 2: Historical data fetch
                           with sentry_sdk.start_span(op="data.fetch") as fetch_span:
                               data_result = await self.fetch_historical_data(contract, **params)
                               fetch_span.set_data("bars_fetched", data_result.get('processed', 0))
                               fetch_span.set_data("task_id", data_result.get('job_id'))
                           
                           # Step 3: Track result
                           results.append({
                               'symbol': symbol,
                               'status': 'success',
                               'data': data_result
                           })
                           span.set_tag("step.status", "success")
                           
                       except Exception as e:
                           results.append({
                               'symbol': symbol,
                               'status': 'error',
                               'error': str(e)
                           })
                           span.set_tag("step.status", "error")
                           span.set_data("error_message", str(e))
               
               # Aggregate results
               successful = sum(1 for r in results if r['status'] == 'success')
               failed = len(results) - successful
               
               transaction.set_data("workflow.symbols_successful", successful)
               transaction.set_data("workflow.symbols_failed", failed)
               transaction.set_tag("workflow.status", "completed")
               
               return {
                   'total_symbols': len(symbols),
                   'successful': successful,
                   'failed': failed,
                   'results': results
               }
   ```

**Deliverables:**
- [ ] Enhanced workflow tracing with step-by-step visibility
- [ ] End-to-end pipeline tracing operational
- [ ] Multi-step operation correlation working

---

## Phase 3: Advanced Monitoring & Metrics (Weeks 5-6)

### Week 5: Custom Trading Metrics

#### Day 21-23: Business KPI Tracking

**Tasks:**
1. **Create Trading Metrics Module**
   ```python
   # File: common/observability/trading_metrics.py
   import sentry_sdk
   from datetime import datetime, timedelta
   from typing import Optional, Dict, Any
   from common.utils.market_hours import is_market_hours, get_market_session
   
   class TradingMetrics:
       @staticmethod
       def track_data_freshness(symbol: str, last_update: datetime, data_type: str = "equity_bars"):
           """Track data freshness for SLA compliance"""
           staleness_seconds = (datetime.utcnow() - last_update).total_seconds()
           staleness_minutes = staleness_seconds / 60
           
           # Set measurements
           sentry_sdk.set_measurement("data.staleness_minutes", staleness_minutes)
           sentry_sdk.set_measurement("data.staleness_seconds", staleness_seconds)
           
           # Set context
           sentry_sdk.set_tag("data.symbol", symbol)
           sentry_sdk.set_tag("data.type", data_type)
           sentry_sdk.set_tag("market.session", get_market_session())
           
           # SLA breach detection
           sla_threshold = 30 if is_market_hours() else 120  # Stricter during market hours
           if staleness_minutes > sla_threshold:
               sentry_sdk.capture_message(
                   f"Data freshness SLA breach: {symbol} ({data_type})",
                   level="warning",
                   extra={
                       "staleness_minutes": staleness_minutes,
                       "sla_threshold": sla_threshold,
                       "market_hours": is_market_hours()
                   }
               )
       
       @staticmethod
       def track_market_data_quality(symbol: str, bars_received: int, expected_bars: int, time_period: str):
           """Monitor market data completeness and quality"""
           completeness = bars_received / expected_bars if expected_bars > 0 else 0
           
           # Set measurements
           sentry_sdk.set_measurement("data.completeness_ratio", completeness)
           sentry_sdk.set_measurement("data.bars_received", bars_received)
           sentry_sdk.set_measurement("data.bars_expected", expected_bars)
           sentry_sdk.set_measurement("data.bars_missing", expected_bars - bars_received)
           
           # Set context
           sentry_sdk.set_tag("data.symbol", symbol)
           sentry_sdk.set_tag("data.time_period", time_period)
           
           # Quality threshold alerts
           if completeness < 0.95:  # Less than 95% complete
               severity = "error" if completeness < 0.90 else "warning"
               sentry_sdk.capture_message(
                   f"Market data quality issue: {symbol}",
                   level=severity,
                   extra={
                       "completeness_ratio": completeness,
                       "bars_received": bars_received,
                       "bars_expected": expected_bars,
                       "time_period": time_period
                   }
               )
       
       @staticmethod
       def track_api_performance(endpoint: str, response_time_ms: float, status_code: int, payload_size: int = 0):
           """Track API endpoint performance metrics"""
           # Set measurements
           sentry_sdk.set_measurement("api.response_time_ms", response_time_ms)
           sentry_sdk.set_measurement("api.payload_size_bytes", payload_size)
           
           # Set tags
           sentry_sdk.set_tag("api.endpoint", endpoint)
           sentry_sdk.set_tag("api.status_code", status_code)
           sentry_sdk.set_tag("api.status_class", f"{status_code // 100}xx")
           
           # Performance SLA monitoring
           if response_time_ms > 2000:  # 2 second SLA
               sentry_sdk.capture_message(
                   f"API performance SLA breach: {endpoint}",
                   level="warning",
                   extra={
                       "response_time_ms": response_time_ms,
                       "sla_threshold_ms": 2000,
                       "status_code": status_code
                   }
               )
       
       @staticmethod
       def track_rate_limiting(service: str, requests_made: int, limit: int, window_seconds: int):
           """Monitor rate limiting compliance and utilization"""
           utilization = requests_made / limit if limit > 0 else 0
           
           # Set measurements  
           sentry_sdk.set_measurement("api.rate_utilization", utilization)
           sentry_sdk.set_measurement("api.requests_made", requests_made)
           sentry_sdk.set_measurement("api.rate_limit", limit)
           
           # Set context
           sentry_sdk.set_tag("api.service", service)
           sentry_sdk.set_tag("api.window_seconds", str(window_seconds))
           
           # Rate limit warnings
           if utilization > 0.90:  # Near rate limit
               sentry_sdk.capture_message(
                   f"Rate limit utilization high: {service}",
                   level="warning",
                   extra={
                       "utilization": utilization,
                       "requests_made": requests_made,
                       "limit": limit,
                       "window_seconds": window_seconds
                   }
               )
       
       @staticmethod
       def track_pipeline_success(pipeline_name: str, symbols_processed: int, symbols_failed: int, duration_seconds: float):
           """Track data pipeline success rates and performance"""
           total_symbols = symbols_processed + symbols_failed
           success_rate = symbols_processed / total_symbols if total_symbols > 0 else 0
           
           # Set measurements
           sentry_sdk.set_measurement("pipeline.success_rate", success_rate)
           sentry_sdk.set_measurement("pipeline.symbols_processed", symbols_processed)
           sentry_sdk.set_measurement("pipeline.symbols_failed", symbols_failed)
           sentry_sdk.set_measurement("pipeline.duration_seconds", duration_seconds)
           sentry_sdk.set_measurement("pipeline.throughput_symbols_per_second", total_symbols / duration_seconds if duration_seconds > 0 else 0)
           
           # Set context
           sentry_sdk.set_tag("pipeline.name", pipeline_name)
           sentry_sdk.set_tag("pipeline.status", "success" if success_rate > 0.95 else "degraded")
           
           # Pipeline health alerts
           if success_rate < 0.99:  # Less than 99% success
               severity = "error" if success_rate < 0.95 else "warning"
               sentry_sdk.capture_message(
                   f"Pipeline success rate degraded: {pipeline_name}",
                   level=severity,
                   extra={
                       "success_rate": success_rate,
                       "symbols_processed": symbols_processed,
                       "symbols_failed": symbols_failed,
                       "duration_seconds": duration_seconds
                   }
               )
   ```

2. **Integrate Metrics into Repositories**
   ```python
   # File: common/repositories/equity_bars.py (enhancement)
   from common.observability.trading_metrics import TradingMetrics
   
   class EquityBarRepository(TracedBaseRepository):
       def save(self, df: pd.DataFrame, symbol: str, **kwargs):
           result = super().save(df, symbol=symbol, **kwargs)
           
           # Track data quality metrics
           TradingMetrics.track_market_data_quality(
               symbol=symbol,
               bars_received=len(df),
               expected_bars=self._calculate_expected_bars(df),
               time_period=kwargs.get('bar_size', 'unknown')
           )
           
           return result
   ```

**Deliverables:**
- [ ] Comprehensive trading metrics module
- [ ] Business KPI tracking integrated into repositories
- [ ] Data quality monitoring operational

#### Day 24-25: Frontend Error Tracking

**Tasks:**
1. **Next.js Sentry Integration**
   ```javascript
   // File: frontend-nextjs/lib/sentry.js
   import * as Sentry from "@sentry/nextjs";
   
   Sentry.init({
     dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
     environment: process.env.NODE_ENV,
     
     tracesSampleRate: process.env.NODE_ENV === "production" ? 0.1 : 1.0,
     
     integrations: [
       new Sentry.BrowserTracing({
         routingInstrumentation: Sentry.reactRouterV6Instrumentation(
           React.useEffect,
           useLocation,
           useNavigationType,
           createRoutesFromChildren,
           matchRoutes
         ),
       }),
       new Sentry.Replay({
         maskAllText: true,  // Privacy protection
         blockAllMedia: true,
       }),
     ],
     
     replaysSessionSampleRate: 0.1,
     replaysOnErrorSampleRate: 1.0,
     
     beforeSend(event) {
       // Filter trading-specific frontend errors
       if (event.exception) {
         const error = event.exception.values[0];
         
         // API connection errors
         if (error.type === "NetworkError" || error.type === "TypeError") {
           event.fingerprint = ["frontend", "network", error.type];
         }
         
         // Data loading errors
         if (error.value?.includes("Failed to fetch")) {
           event.fingerprint = ["frontend", "api-fetch", event.request?.url];
         }
         
         // Chart rendering errors
         if (error.value?.includes("chart") || error.value?.includes("visualization")) {
           event.fingerprint = ["frontend", "visualization", error.type];
         }
       }
       
       return event;
     }
   });
   ```

2. **Trading-Specific Frontend Hooks**
   ```javascript
   // File: frontend-nextjs/hooks/useTradingData.js
   import { useQuery } from '@tanstack/react-query';
   import * as Sentry from '@sentry/nextjs';
   
   export function useTradingData(symbol, options = {}) {
     return useQuery({
       queryKey: ['trading-data', symbol, options],
       queryFn: async () => {
         // Set Sentry context for trading data requests
         Sentry.setContext("trading", {
           symbol,
           timestamp: new Date().toISOString(),
           component: "TradingDataHook",
           options
         });
         
         Sentry.addBreadcrumb({
           message: `Requesting trading data for ${symbol}`,
           category: "data",
           level: "info",
           data: { symbol, options }
         });
         
         try {
           const startTime = performance.now();
           const data = await fetchTradingData(symbol, options);
           const duration = performance.now() - startTime;
           
           // Track successful data loads
           Sentry.setMeasurement("frontend.data_load_time", duration);
           Sentry.addBreadcrumb({
             message: `Trading data loaded for ${symbol}`,
             category: "data", 
             level: "info",
             data: { 
               symbol, 
               recordCount: data.length,
               loadTime: duration
             }
           });
           
           return data;
           
         } catch (error) {
           // Enhanced error context
           Sentry.setContext("error_context", {
             symbol,
             dataType: "trading",
             timestamp: new Date().toISOString(),
             options
           });
           
           Sentry.captureException(error, {
             tags: {
               component: "TradingDataHook",
               symbol,
               operation: "data_fetch"
             }
           });
           
           throw error;
         }
       },
       retry: (failureCount, error) => {
         if (failureCount >= 3) {
           Sentry.captureMessage(
             `Max retries exceeded for trading data: ${symbol}`,
             "warning",
             {
               tags: { symbol, component: "TradingDataHook" },
               extra: { failureCount, error: error.message }
             }
           );
           return false;
         }
         return true;
       },
       staleTime: 5 * 60 * 1000, // 5 minutes
       cacheTime: 10 * 60 * 1000, // 10 minutes
     });
   }
   ```

**Deliverables:**
- [ ] Next.js Sentry integration with session replay
- [ ] Trading-specific error tracking in frontend
- [ ] Performance monitoring for data loading

### Week 6: Alerting & Dashboards

#### Day 26-28: Alert Configuration

**Tasks:**
1. **Business-Critical Alert Rules**
   ```python
   # File: common/observability/alert_rules.py
   from typing import Dict, Any, List
   
   class TradingAlertRules:
       """Sentry alert rule configurations for trading platform"""
       
       MARKET_DATA_OUTAGE = {
           'name': 'Market Data Outage',
           'condition': {
               'dataset': 'events',
               'query': 'level:fatal AND message:"market-data"',
               'aggregation': 'count()',
               'time_window': 10,  # minutes
               'threshold': 1
           },
           'severity': 'high',
           'actions': {
               'slack': '#alerts-critical',
               'email': 'oncall@trading.com',
               'webhook': 'https://api.pagerduty.com/incidents'
           }
       }
       
       IB_API_CONNECTION_LOSS = {
           'name': 'IB API Connection Loss',
           'condition': {
               'dataset': 'events',
               'query': 'level:error AND (message:"connection" OR message:"disconnect")',
               'aggregation': 'count()',
               'time_window': 5,  # minutes
               'threshold': 3
           },
           'severity': 'medium',
           'actions': {
               'slack': '#alerts-trading',
               'email': 'trading-team@trading.com'
           }
       }
       
       HIGH_ERROR_RATE = {
           'name': 'High API Error Rate',
           'condition': {
               'dataset': 'transactions',
               'query': 'http.status_code:[500 TO 599]',
               'aggregation': 'failure_rate()',
               'time_window': 15,  # minutes
               'threshold': 0.05  # 5% error rate
           },
           'severity': 'medium',
           'actions': {
               'slack': '#alerts-engineering',
               'email': 'engineering@trading.com'
           }
       }
       
       DATA_PIPELINE_FAILURES = {
           'name': 'Data Pipeline High Failure Rate',
           'condition': {
               'dataset': 'events',
               'query': 'level:error AND pipeline.success_rate:[* TO 0.95]',
               'aggregation': 'count()',
               'time_window': 30,  # minutes
               'threshold': 1
           },
           'severity': 'medium',
           'actions': {
               'slack': '#alerts-data',
               'email': 'data-team@trading.com'
           }
       }
       
       RATE_LIMIT_NEAR_BREACH = {
           'name': 'Rate Limit Near Breach',
           'condition': {
               'dataset': 'events',
               'query': 'api.rate_utilization:[0.9 TO *]',
               'aggregation': 'count()',
               'time_window': 5,  # minutes
               'threshold': 5
           },
           'severity': 'low',
           'actions': {
               'slack': '#alerts-api'
           }
       }
   ```

2. **Alert Setup Script**
   ```python
   # File: ops/scripts/setup_sentry_alerts.py
   import requests
   from typing import Dict, Any
   from app_settings.settings import get_settings
   
   class SentryAlertSetup:
       def __init__(self):
           self.settings = get_settings()
           self.auth_token = self.settings.SENTRY_AUTH_TOKEN
           self.org_slug = self.settings.SENTRY_ORG_SLUG
           self.base_url = "https://sentry.io/api/0"
       
       def create_alert_rule(self, project_slug: str, rule_config: Dict[str, Any]):
           """Create Sentry alert rule via API"""
           url = f"{self.base_url}/projects/{self.org_slug}/{project_slug}/alert-rules/"
           
           headers = {
               'Authorization': f'Bearer {self.auth_token}',
               'Content-Type': 'application/json'
           }
           
           payload = {
               'name': rule_config['name'],
               'conditions': [rule_config['condition']],
               'actions': self._format_actions(rule_config['actions']),
               'actionMatch': 'all',
               'frequency': 60,  # Check every minute
               'environment': self.settings.APP_ENV
           }
           
           response = requests.post(url, json=payload, headers=headers)
           response.raise_for_status()
           
           return response.json()
       
       def setup_all_alerts(self):
           """Set up all alert rules across projects"""
           from common.observability.alert_rules import TradingAlertRules
           
           projects = [
               'trading-platform-calendar-api',
               'trading-platform-equity-api',
               'trading-platform-orchestrator-api',
               # ... other projects
           ]
           
           alert_rules = [
               TradingAlertRules.MARKET_DATA_OUTAGE,
               TradingAlertRules.IB_API_CONNECTION_LOSS,
               TradingAlertRules.HIGH_ERROR_RATE,
               TradingAlertRules.DATA_PIPELINE_FAILURES,
               TradingAlertRules.RATE_LIMIT_NEAR_BREACH
           ]
           
           for project in projects:
               for rule in alert_rules:
                   try:
                       result = self.create_alert_rule(project, rule)
                       print(f"Created alert '{rule['name']}' for {project}")
                   except Exception as e:
                       print(f"Failed to create alert '{rule['name']}' for {project}: {e}")
   ```

**Deliverables:**
- [ ] Business-critical alert rules defined
- [ ] Alert setup automation script
- [ ] Notification channels configured

#### Day 29-30: Custom Dashboards

**Tasks:**
1. **Create Trading Performance Dashboard**
   ```python
   # File: ops/scripts/setup_sentry_dashboards.py
   import requests
   from typing import Dict, Any, List
   
   class TradingDashboards:
       def __init__(self):
           # Dashboard configuration
           pass
       
       TRADING_PERFORMANCE_DASHBOARD = {
           'title': 'Trading Platform Performance',
           'widgets': [
               {
                   'title': 'API Response Times (P95)',
                   'displayType': 'line',
                   'query': 'p95(api.response_time_ms)',
                   'yAxis': 'Response Time (ms)'
               },
               {
                   'title': 'Error Rate by Service',
                   'displayType': 'bar',
                   'query': 'failure_rate() BY service',
                   'yAxis': 'Error Rate (%)'
               },
               {
                   'title': 'Data Freshness by Symbol',
                   'displayType': 'table',
                   'query': 'avg(data.staleness_minutes) BY trading.symbol',
                   'yAxis': 'Minutes Stale'
               },
               {
                   'title': 'Pipeline Success Rates',
                   'displayType': 'big_number',
                   'query': 'avg(pipeline.success_rate)',
                   'yAxis': 'Success Rate'
               },
               {
                   'title': 'Rate Limit Utilization',
                   'displayType': 'line',
                   'query': 'avg(api.rate_utilization) BY api.service',
                   'yAxis': 'Utilization (%)'
               }
           ]
       }
       
       BUSINESS_INTELLIGENCE_DASHBOARD = {
           'title': 'Trading Business Intelligence',
           'widgets': [
               {
                   'title': 'Daily Data Volume',
                   'displayType': 'area',
                   'query': 'sum(data.bars_received)',
                   'yAxis': 'Bars Processed'
               },
               {
                   'title': 'Most Active Symbols',
                   'displayType': 'table',
                   'query': 'count() BY trading.symbol',
                   'yAxis': 'Request Count'
               },
               {
                   'title': 'Market Hours vs After Hours Activity',
                   'displayType': 'pie',
                   'query': 'count() BY market.session',
                   'yAxis': 'Request Distribution'
               },
               {
                   'title': 'Data Quality Score',
                   'displayType': 'big_number',
                   'query': 'avg(data.completeness_ratio)',
                   'yAxis': 'Quality Score'
               }
           ]
       }
   ```

**Deliverables:**
- [ ] Trading performance dashboard created
- [ ] Business intelligence dashboard configured  
- [ ] Custom metrics visualization operational

---

## Phase 4: Testing & Production Deployment (Week 7)

### Week 7: Final Testing & Go-Live

#### Day 31-33: Load Testing & Performance

**Tasks:**
1. **Observability Load Testing**
   ```python
   # File: tests/load/test_observability_overhead.py
   import asyncio
   import time
   import statistics
   from concurrent.futures import ThreadPoolExecutor
   from common.observability.sentry_config import configure_sentry
   from services.equity_api.main import app
   
   async def test_observability_overhead():
       """Measure performance impact of Sentry integration"""
       
       # Test without Sentry
       baseline_times = await run_load_test(sentry_enabled=False)
       
       # Test with Sentry
       configure_sentry("equity-api")
       sentry_times = await run_load_test(sentry_enabled=True)
       
       # Calculate overhead
       baseline_avg = statistics.mean(baseline_times)
       sentry_avg = statistics.mean(sentry_times)
       overhead_percent = ((sentry_avg - baseline_avg) / baseline_avg) * 100
       
       print(f"Baseline average: {baseline_avg:.2f}ms")
       print(f"Sentry average: {sentry_avg:.2f}ms") 
       print(f"Overhead: {overhead_percent:.2f}%")
       
       # Assert acceptable overhead (< 10%)
       assert overhead_percent < 10, f"Sentry overhead too high: {overhead_percent:.2f}%"
   
   async def run_load_test(sentry_enabled: bool, num_requests: int = 1000):
       """Run load test and return response times"""
       response_times = []
       
       async def make_request():
           start_time = time.time()
           # Simulate API request
           await asyncio.sleep(0.01)  # 10ms base response time
           end_time = time.time()
           return (end_time - start_time) * 1000  # Convert to ms
       
       # Run concurrent requests
       tasks = [make_request() for _ in range(num_requests)]
       response_times = await asyncio.gather(*tasks)
       
       return response_times
   ```

2. **Alert Testing**
   ```python
   # File: tests/integration/test_alert_system.py
   import sentry_sdk
   import pytest
   from common.observability.trading_metrics import TradingMetrics
   
   class TestAlertSystem:
       def test_market_data_outage_alert(self):
           """Test that market data outage triggers alert"""
           # Simulate market data outage
           sentry_sdk.capture_message(
               "Market data outage detected",
               level="fatal",
               fingerprint=['market-data', 'outage']
           )
           
           # Alert should be triggered (manual verification in Sentry)
           
       def test_high_error_rate_alert(self):
           """Test high error rate alert threshold"""
           # Simulate multiple errors
           for i in range(10):
               try:
                   raise Exception(f"Test error {i}")
               except Exception as e:
                   sentry_sdk.capture_exception(e)
           
           # High error rate alert should trigger
           
       def test_data_freshness_alert(self):
           """Test data freshness SLA breach"""
           from datetime import datetime, timedelta
           
           # Simulate stale data
           old_timestamp = datetime.utcnow() - timedelta(minutes=45)
           TradingMetrics.track_data_freshness("AAPL", old_timestamp)
           
           # Freshness alert should trigger
   ```

**Deliverables:**
- [ ] Performance overhead measurements completed
- [ ] Load testing with acceptable impact verified
- [ ] Alert system tested and validated

#### Day 34-35: Production Deployment

**Tasks:**
1. **Production Configuration**
   ```bash
   # Environment variables for production
   export SENTRY_DSN="https://xxx@xxx.ingest.sentry.io/production"
   export SENTRY_TRACES_SAMPLE_RATE="0.1"
   export SENTRY_PROFILES_SAMPLE_RATE="0.05"
   export OBSERVABILITY_ENABLED="true"
   ```

2. **Gradual Rollout Plan**
   ```python
   # File: ops/scripts/gradual_rollout.py
   
   ROLLOUT_PHASES = [
       {
           'phase': 1,
           'services': ['calendar-api'],  # Start with lowest traffic service
           'duration': '24 hours',
           'monitoring_focus': ['basic error tracking', 'performance overhead']
       },
       {
           'phase': 2, 
           'services': ['calendar-api', 'equity-api'],
           'duration': '48 hours',
           'monitoring_focus': ['distributed tracing', 'cross-service correlation']
       },
       {
           'phase': 3,
           'services': ['calendar-api', 'equity-api', 'orchestrator-api'],
           'duration': '48 hours', 
           'monitoring_focus': ['celery task tracing', 'workflow visibility']
       },
       {
           'phase': 4,
           'services': ['all-backend-services'],
           'duration': '72 hours',
           'monitoring_focus': ['full system observability', 'alert tuning']
       },
       {
           'phase': 5,
           'services': ['all-services', 'frontend'],
           'duration': 'ongoing',
           'monitoring_focus': ['business metrics', 'optimization']
       }
   ]
   ```

3. **Rollback Procedures**
   ```bash
   # File: ops/scripts/rollback_observability.sh
   
   #!/bin/bash
   # Emergency rollback script for Sentry integration
   
   echo "Disabling Sentry integration..."
   
   # Set environment variable to disable Sentry
   export OBSERVABILITY_ENABLED="false"
   
   # Restart services without Sentry
   make services-restart
   
   # Verify services are healthy
   make health-check
   
   echo "Sentry integration disabled. Services running with local logging only."
   ```

**Deliverables:**
- [ ] Production environment configured
- [ ] Gradual rollout plan executed
- [ ] Rollback procedures tested and documented
- [ ] Full observability platform operational

---

## Success Metrics & Validation

### Technical Validation Checklist

#### Phase 1 Validation
- [ ] **Error Tracking**: All 6 services reporting errors to Sentry
- [ ] **Context Enhancement**: Trading-specific tags and context visible in Sentry  
- [ ] **Error Classification**: Errors properly categorized and fingerprinted
- [ ] **Integration Health**: FastAPI middleware capturing request context

#### Phase 2 Validation  
- [ ] **Cross-Service Tracing**: HTTP calls between services showing in traces
- [ ] **Database Tracing**: Repository operations visible in performance monitoring
- [ ] **Celery Integration**: Task execution traces with business context
- [ ] **Workflow Visibility**: End-to-end pipeline tracing operational

#### Phase 3 Validation
- [ ] **Custom Metrics**: Trading KPIs flowing to Sentry measurements
- [ ] **Frontend Monitoring**: Next.js errors and performance tracked
- [ ] **Alert System**: Business-critical alerts configured and tested
- [ ] **Dashboard Creation**: Trading performance dashboards operational

#### Phase 4 Validation
- [ ] **Performance Impact**: < 5% performance overhead measured
- [ ] **Production Stability**: 48 hours production operation without issues
- [ ] **Alert Accuracy**: < 10% false positive rate achieved
- [ ] **Team Adoption**: Development team trained and using dashboards

### Business Impact Measurements

#### Week 4 Baseline (End of Phase 2)
- [ ] **MTTD Baseline**: Current mean time to detect issues
- [ ] **MTTR Baseline**: Current mean time to resolve incidents  
- [ ] **Manual Monitoring Time**: Hours spent on manual system monitoring
- [ ] **Incident Response Time**: Time from issue occurrence to team notification

#### Week 7 Final Measurements (End of Phase 4)
- [ ] **MTTD Improvement**: Target < 5 minutes for critical errors
- [ ] **MTTR Improvement**: Target < 30 minutes for P0 incidents
- [ ] **Monitoring Efficiency**: 80% reduction in manual monitoring time
- [ ] **Proactive Detection**: > 70% of issues detected before user impact

---

## Risk Mitigation & Contingency Plans

### High-Impact Risk Scenarios

#### Scenario 1: Sentry Service Outage
**Mitigation**:
- Graceful degradation built into SDK configuration
- Local logging continues uninterrupted
- Health checks independent of Sentry status
- 24-hour RTO with backup observability tools

#### Scenario 2: Performance Degradation
**Mitigation**:
- Real-time performance monitoring during rollout
- Automatic sampling rate reduction at 5% overhead threshold
- Feature flags for instant Sentry disable
- Pre-configured rollback procedures

#### Scenario 3: Alert Fatigue
**Mitigation**:
- Conservative alerting thresholds during initial deployment
- 2-week alert tuning period with daily reviews
- Alert escalation rules to prevent spam
- Business-critical vs informational alert separation

#### Scenario 4: Security/Compliance Issues  
**Mitigation**:
- PII filtering review by security team
- Data residency compliance verified
- Access control audit before production deployment
- Privacy impact assessment completed

### Rollback Triggers

**Immediate Rollback Required If**:
- > 10% performance degradation observed
- Critical system functionality impaired
- Data privacy breach detected
- > 50% false positive alert rate

**Rollback Procedure**:
1. Set `OBSERVABILITY_ENABLED=false` environment variable
2. Restart all services (< 30 seconds downtime)
3. Verify system health with existing monitoring
4. Investigate issue and plan remediation

---

## Team Readiness & Training

### Training Schedule

#### Week 1: Development Team Training (2 hours)
**Topics**:
- Sentry UI navigation and basic workflows
- Reading error reports and stack traces
- Understanding distributed traces
- Using context and tags for debugging

#### Week 4: Advanced Training (2 hours)
**Topics**:
- Creating custom metrics and measurements
- Alert rule configuration
- Dashboard customization
- Performance monitoring interpretation

#### Week 6: Incident Response Training (1 hour)
**Topics**:
- Alert escalation procedures
- Using Sentry during incident response
- Correlation ID tracing across services
- Business impact assessment

### Documentation Deliverables

#### Developer Documentation
- [ ] **Sentry Integration Guide**: How to add observability to new services
- [ ] **Error Handling Best Practices**: Trading-specific error patterns  
- [ ] **Custom Metrics Guide**: Creating and tracking business KPIs
- [ ] **Troubleshooting Playbook**: Common observability issues and solutions

#### Operations Documentation
- [ ] **Incident Response Runbooks**: Step-by-step incident handling with Sentry
- [ ] **Alert Configuration Guide**: Managing and tuning alert rules
- [ ] **Dashboard Management**: Creating and maintaining custom dashboards
- [ ] **Performance Monitoring Guide**: Interpreting performance metrics

---

## Budget & Resource Planning

### Cost Breakdown

#### Sentry SaaS Subscription
- **Organization Plan**: $80/month
- **Estimated Usage**: 200K errors/month, 100M performance units/month
- **Annual Cost**: $960 + usage overages

#### Development Resources
- **Phase 1-2**: 40 hours (1 developer, 1 week)
- **Phase 3-4**: 80 hours (2 developers, 1 week each)
- **Testing & Deployment**: 40 hours (1 developer, 1 week)
- **Total Development**: 160 hours

#### Ongoing Maintenance
- **Alert Tuning**: 4 hours/month
- **Dashboard Updates**: 2 hours/month  
- **Performance Review**: 4 hours/month
- **Total Ongoing**: 10 hours/month

### ROI Calculation

#### Current Costs (Manual Monitoring)
- **Incident Detection**: 2 hours/week × 52 weeks = 104 hours/year
- **System Monitoring**: 5 hours/week × 52 weeks = 260 hours/year
- **Troubleshooting**: 8 hours/incident × 24 incidents/year = 192 hours/year
- **Total Current**: 556 hours/year

#### Expected Savings with Observability
- **80% reduction in manual monitoring**: 208 hours saved/year
- **50% faster incident resolution**: 96 hours saved/year  
- **Proactive issue detection**: 48 hours saved/year
- **Total Savings**: 352 hours/year

#### Net ROI
- **Annual Savings**: 352 hours × $150/hour = $52,800
- **Annual Costs**: $960 (Sentry) + 120 hours × $150/hour (maintenance) = $18,960
- **Net Annual Benefit**: $33,840
- **ROI**: 178% return on investment

---

## Implementation Timeline Summary

| Phase | Duration | Key Deliverables | Success Criteria |
|-------|----------|------------------|------------------|
| **Phase 1: Foundation** | Weeks 1-2 | Core SDK integration, error tracking, context enhancement | All services reporting to Sentry with trading context |
| **Phase 2: Tracing** | Weeks 3-4 | Distributed tracing, Celery integration, workflow visibility | End-to-end trace visibility across services |
| **Phase 3: Advanced** | Weeks 5-6 | Custom metrics, frontend monitoring, alerting, dashboards | Business KPIs tracked, alerts operational |
| **Phase 4: Production** | Week 7 | Load testing, gradual rollout, validation, documentation | Production deployment with < 5% performance impact |

**Total Timeline**: 7 weeks
**Go-Live Date**: End of Week 7
**Full Operational**: Week 8

---

*This implementation plan provides a comprehensive, step-by-step roadmap for deploying world-class observability to our trading platform. The gradual, risk-managed approach ensures successful adoption while maintaining system reliability and performance.*