# Observability Specification - Trading Platform
*Enhanced Error Tracking, Logging, and Distributed Tracing with Sentry*

## Executive Summary

This specification outlines the implementation of comprehensive observability for our trading platform using **Sentry as the primary observability platform**. The platform already has sophisticated logging and monitoring infrastructure, making it exceptionally well-prepared for Sentry integration.

**Key Benefits:**
- 🎯 **Centralized Error Tracking**: Real-time error detection across 6 microservices
- 📊 **Distributed Tracing**: End-to-end transaction visibility across services and Celery tasks  
- ⚡ **Performance Monitoring**: API endpoint performance, database queries, and external API calls
- 🔔 **Proactive Alerting**: Business-critical error notifications and SLA monitoring
- 📈 **Business Intelligence**: Trading-specific metrics and operational insights

## Current Architecture Analysis

### **Microservices Inventory**

| Service | Port | Purpose | Key Observability Needs |
|---------|------|---------|-------------------------|
| **Calendar API** | 9001 | Earnings calendar management | Rate limiting, data freshness alerts |
| **Equity API** | 9002 | Historical equity data retrieval | Query performance, cache hit rates |
| **Option History API** | 9003 | Option bar data serving | Large dataset performance |
| **Option Chain API** | 9004 | Real-time option chain snapshots | Market data reliability, latency |
| **Orchestrator API** | 9005 | Async workflow coordination | Task success rates, workflow tracing |
| **IB API Service** | 9010 | Interactive Brokers integration | Connection stability, rate limiting |

### **Celery Task Architecture**

```
Queue Structure:
├── contracts (Contract management)
├── historical_data (Data backfill)  
├── option_chains (Option data processing)
├── equity_bars (Equity data ingestion)
└── default (General async tasks)
```

**Critical Monitoring Points:**
- Task success/failure rates per queue
- Rate limiting compliance (2 req/sec for IB API)
- Data pipeline integrity and correlation ID tracing
- Memory usage in large dataset processing

### **Existing Observability Infrastructure** ⭐

**Strengths to Leverage:**
- ✅ **Structured JSON Logging** with correlation IDs
- ✅ **Circuit Breaker Patterns** with reliability metrics
- ✅ **Performance Decorators** for timing critical operations  
- ✅ **Thread-Local Context** for distributed request tracking
- ✅ **Trading-Specific Exception Hierarchy** for business logic errors
- ✅ **Real-time Dashboard** at http://localhost:9090

## Sentry Integration Strategy

### **Phase 1: Core Error Tracking (Week 1-2)**

#### **1.1 Sentry SDK Installation & Configuration**

**Dependencies:**
```toml
# Add to pyproject.toml [project.dependencies]
"sentry-sdk[fastapi,celery]>=1.40.0",
```

**Core Configuration:**
```python
# common/observability/sentry_config.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

def configure_sentry(environment: str, service_name: str):
    sentry_sdk.init(
        dsn=get_sentry_dsn(),
        environment=environment,
        traces_sample_rate=get_trace_sample_rate(environment),
        profiles_sample_rate=get_profile_sample_rate(environment),
        send_default_pii=False,  # Trading compliance requirement
        
        # Service identification
        server_name=service_name,
        release=get_app_version(),
        
        # Integrations
        integrations=[
            FastApiIntegration(
                auto_enabling_integrations=True,
                transaction_style="endpoint",
                failed_request_status_codes={*range(500, 600)}
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
        
        # Trading-specific configuration
        before_send=trading_error_filter,
        before_send_transaction=trading_transaction_filter
    )
```

**Service-Specific Integration Points:**

```python
# services/calendar_api/main.py - Calendar Service
configure_sentry(environment="production", service_name="calendar-api")

# services/equity_api/main.py - Equity Service  
configure_sentry(environment="production", service_name="equity-api")

# services/orchestrator_api/main.py - Orchestrator
configure_sentry(environment="production", service_name="orchestrator-api")

# Similar for option_history_api, option_chain_api, ib_api
```

#### **1.2 Enhanced Error Context**

**Trading-Specific Context:**
```python
# common/observability/trading_context.py
class TradingContextManager:
    @staticmethod
    def set_trading_context(
        symbol: Optional[str] = None,
        market_session: Optional[str] = None,
        data_source: Optional[str] = None,
        correlation_id: Optional[str] = None
    ):
        with sentry_sdk.configure_scope() as scope:
            if symbol:
                scope.set_tag("trading.symbol", symbol)
            if market_session:
                scope.set_tag("trading.session", market_session)
            if data_source:
                scope.set_tag("trading.data_source", data_source)
            if correlation_id:
                scope.set_context("correlation", {"id": correlation_id})
            
            # Business context
            scope.set_context("trading", {
                "timestamp": datetime.utcnow().isoformat(),
                "market_hours": is_market_hours(),
                "service_health": get_service_health_status()
            })
```

**Error Classification:**
```python
# common/observability/error_classification.py
def trading_error_filter(event, hint):
    """Filter and enhance trading-specific errors"""
    
    # Business-critical errors
    if is_market_data_error(event):
        event['level'] = 'fatal'
        event['fingerprint'] = ['market-data', event.get('exception', {}).get('type')]
    
    # Rate limiting errors (expected, lower severity)
    elif is_rate_limit_error(event):
        event['level'] = 'warning'
        event['fingerprint'] = ['rate-limit', event.get('transaction')]
        
    # Data integrity errors
    elif is_data_integrity_error(event):
        event['level'] = 'error'
        event['fingerprint'] = ['data-integrity', event.get('contexts', {}).get('trading', {}).get('symbol')]
    
    return event
```

#### **1.3 FastAPI Integration**

**Middleware Enhancement:**
```python
# common/observability/middleware.py
class TradingObservabilityMiddleware:
    async def __call__(self, request: Request, call_next):
        # Extract trading context from request
        symbol = request.path_params.get('symbol')
        correlation_id = request.headers.get('X-Correlation-ID') or generate_correlation_id()
        
        # Set Sentry context
        TradingContextManager.set_trading_context(
            symbol=symbol,
            correlation_id=correlation_id,
            data_source="api_request"
        )
        
        # Add to existing logging context
        with trading_context(correlation_id=correlation_id, symbol=symbol):
            response = await call_next(request)
            
        return response
```

### **Phase 2: Distributed Tracing (Week 3-4)**

#### **2.1 Cross-Service Trace Propagation**

**HTTP Client Integration:**
```python
# common/utils/http_client.py - Enhanced with Sentry tracing
class TracedHttpClient:
    def __init__(self):
        self.client = httpx.AsyncClient()
    
    async def request(self, method: str, url: str, **kwargs):
        with sentry_sdk.start_transaction(
            op="http.client", 
            name=f"{method.upper()} {url}"
        ) as transaction:
            
            # Propagate trace headers
            headers = kwargs.get('headers', {})
            trace_headers = sentry_sdk.Hub.current.trace_propagation_meta
            headers.update(trace_headers)
            
            response = await self.client.request(method, url, headers=headers, **kwargs)
            
            transaction.set_tag("http.status_code", response.status_code)
            transaction.set_data("response_size", len(response.content))
            
            return response
```

**Database Query Tracing:**
```python
# common/repositories/base.py - Enhanced repository tracing
class TracedBaseRepository:
    def save(self, df: pd.DataFrame, **kwargs):
        with sentry_sdk.start_span(
            op="db.query", 
            description=f"Save to {self.__class__.__name__}"
        ) as span:
            span.set_data("rows_count", len(df))
            span.set_data("columns", list(df.columns))
            span.set_tag("repository.type", self.repository_type)
            
            result = super().save(df, **kwargs)
            
            span.set_data("files_written", len(result.get('files', [])))
            return result
```

#### **2.2 Celery Task Distributed Tracing**

**Task Chain Visibility:**
```python
# ops/tasks/base.py - Enhanced Celery task tracing
from celery import Task
from sentry_sdk.integrations.celery import CeleryIntegration

class TracedTradingTask(Task):
    def __call__(self, *args, **kwargs):
        # Extract business context
        symbol = kwargs.get('symbol') or args[0] if args else None
        
        with sentry_sdk.start_transaction(
            op="celery.task",
            name=f"{self.name}",
            sampled=True
        ) as transaction:
            
            TradingContextManager.set_trading_context(
                symbol=symbol,
                data_source="celery_task",
                correlation_id=self.request.id
            )
            
            transaction.set_tag("celery.task_name", self.name)
            transaction.set_tag("celery.queue", self.request.delivery_info.get('routing_key'))
            
            try:
                result = super().__call__(*args, **kwargs)
                transaction.set_tag("task.status", "success")
                return result
            except Exception as exc:
                transaction.set_tag("task.status", "failure")
                sentry_sdk.capture_exception(exc)
                raise
```

**Workflow Correlation:**
```python
# services/orchestrator_api/workflows.py - Workflow tracing
class TracedOrchestrator:
    async def execute_data_pipeline(self, symbols: List[str]):
        with sentry_sdk.start_transaction(
            op="workflow.data_pipeline",
            name="Historical Data Pipeline"
        ) as transaction:
            
            transaction.set_data("symbols_count", len(symbols))
            transaction.set_tag("workflow.type", "historical_data")
            
            for symbol in symbols:
                with sentry_sdk.start_span(
                    op="workflow.step",
                    description=f"Process {symbol}"
                ) as span:
                    
                    span.set_tag("trading.symbol", symbol)
                    
                    # Step 1: Contract resolution
                    with sentry_sdk.start_span(op="contract.resolve") as contract_span:
                        contract = await self.resolve_contract(symbol)
                    
                    # Step 2: Historical data fetch
                    with sentry_sdk.start_span(op="data.fetch") as fetch_span:
                        data = await self.fetch_historical_data(contract)
                        fetch_span.set_data("bars_fetched", len(data))
                    
                    # Step 3: Data persistence
                    with sentry_sdk.start_span(op="data.persist") as persist_span:
                        result = await self.save_data(data, symbol)
                        persist_span.set_data("files_saved", result.saved)
```

### **Phase 3: Advanced Monitoring & Business Intelligence (Week 5-6)**

#### **3.1 Custom Metrics & Business KPIs**

**Trading Performance Metrics:**
```python
# common/observability/metrics.py
class TradingMetrics:
    @staticmethod
    def track_data_freshness(symbol: str, last_update: datetime):
        """Track data freshness for compliance"""
        staleness_minutes = (datetime.utcnow() - last_update).total_seconds() / 60
        
        sentry_sdk.set_measurement("data.staleness_minutes", staleness_minutes)
        sentry_sdk.set_tag("data.symbol", symbol)
        
        if staleness_minutes > 30:  # SLA breach
            sentry_sdk.capture_message(
                f"Data staleness SLA breach: {symbol}",
                level="warning"
            )
    
    @staticmethod
    def track_market_data_quality(symbol: str, bars_count: int, expected_count: int):
        """Monitor market data completeness"""
        completeness = bars_count / expected_count if expected_count > 0 else 0
        
        sentry_sdk.set_measurement("data.completeness_ratio", completeness)
        sentry_sdk.set_measurement("data.bars_received", bars_count)
        
        if completeness < 0.95:  # Quality threshold
            sentry_sdk.capture_message(
                f"Market data quality issue: {symbol} ({completeness:.1%} complete)",
                level="warning"
            )
    
    @staticmethod 
    def track_api_rate_limiting(service: str, requests_made: int, limit: int):
        """Monitor rate limiting compliance"""
        utilization = requests_made / limit
        
        sentry_sdk.set_measurement("api.rate_utilization", utilization)
        sentry_sdk.set_tag("api.service", service)
        
        if utilization > 0.90:  # Near rate limit
            sentry_sdk.capture_message(
                f"Rate limit utilization high: {service} ({utilization:.1%})",
                level="warning"
            )
```

#### **3.2 Frontend Error Tracking**

**Next.js Integration:**
```javascript
// frontend-nextjs/lib/sentry.js
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
  ],
  
  beforeSend(event) {
    // Filter trading-specific frontend errors
    if (event.exception) {
      const error = event.exception.values[0];
      
      // API connection errors
      if (error.type === "NetworkError") {
        event.fingerprint = ["frontend", "network", error.value];
      }
      
      // Data loading errors  
      if (error.value?.includes("Failed to fetch")) {
        event.fingerprint = ["frontend", "data-fetch", event.request?.url];
      }
    }
    
    return event;
  }
});
```

**Trading UI Error Context:**
```javascript
// frontend-nextjs/hooks/useTradingData.js
export function useTradingData(symbol) {
  return useQuery({
    queryKey: ['trading-data', symbol],
    queryFn: async () => {
      // Set Sentry context for trading data requests
      Sentry.setContext("trading", {
        symbol,
        timestamp: new Date().toISOString(),
        component: "TradingDataHook"
      });
      
      try {
        const data = await fetchTradingData(symbol);
        
        // Track successful data loads
        Sentry.addBreadcrumb({
          message: `Trading data loaded for ${symbol}`,
          category: "data",
          level: "info",
          data: { symbol, recordCount: data.length }
        });
        
        return data;
      } catch (error) {
        // Enhanced error context for trading data failures
        Sentry.setContext("error_context", {
          symbol,
          dataType: "trading",
          timestamp: new Date().toISOString()
        });
        
        throw error;
      }
    },
    retry: (failureCount, error) => {
      // Custom retry logic with Sentry tracking
      if (failureCount >= 3) {
        Sentry.captureMessage(
          `Max retries exceeded for trading data: ${symbol}`,
          "warning"
        );
        return false;
      }
      return true;
    }
  });
}
```

#### **3.3 Alerting & Notification Strategy**

**Business-Critical Alerts:**

| Alert Category | Condition | Severity | Action |
|----------------|-----------|----------|---------|
| **Market Data Outage** | No data received >10min during market hours | Fatal | Page on-call engineer |
| **IB API Disconnection** | Connection loss >5min | Error | Automatic reconnection + notification |
| **Data Pipeline Failure** | >5% of daily symbols failing | Error | Escalate to data team |
| **Rate Limit Violations** | >95% of rate limit used | Warning | Throttle requests |
| **High Error Rate** | >1% 5xx errors across APIs | Warning | Performance investigation |
| **Memory Usage** | >80% memory usage | Warning | Scale/restart services |

**Alert Configuration:**
```python
# common/observability/alerts.py
class TradingAlertRules:
    MARKET_DATA_OUTAGE = {
        'condition': 'no_data_received_minutes > 10 AND is_market_hours = true',
        'severity': 'fatal',
        'notification_channels': ['pagerduty', 'slack_critical']
    }
    
    IB_API_DISCONNECTION = {
        'condition': 'ib_connection_status = "disconnected" AND duration_minutes > 5',
        'severity': 'error', 
        'notification_channels': ['slack_alerts', 'email_oncall']
    }
    
    HIGH_ERROR_RATE = {
        'condition': 'error_rate_5xx > 1% AND time_window = 5min',
        'severity': 'warning',
        'notification_channels': ['slack_alerts']
    }
```

## Implementation Timeline

### **Week 1-2: Foundation Setup**
- [ ] Sentry project creation and DSN configuration
- [ ] Core SDK integration across 6 microservices  
- [ ] Basic error tracking with trading context
- [ ] FastAPI middleware integration
- [ ] Initial error classification and filtering

### **Week 3-4: Distributed Tracing**
- [ ] Cross-service HTTP trace propagation
- [ ] Celery task tracing and correlation
- [ ] Database operation tracing
- [ ] Workflow visibility in Orchestrator
- [ ] Performance baseline establishment

### **Week 5-6: Advanced Features**
- [ ] Custom trading metrics and KPIs
- [ ] Frontend error tracking (Next.js)
- [ ] Business intelligence dashboards
- [ ] Alert rules and notification setup
- [ ] SLA monitoring and compliance tracking

### **Week 7: Testing & Optimization**
- [ ] Load testing with observability overhead measurement
- [ ] Alert tuning and false positive reduction  
- [ ] Performance optimization
- [ ] Documentation and runbook creation
- [ ] Team training and handover

## Configuration Management

### **Environment-Specific Settings**

```python
# config/environments/production.py
SENTRY_CONFIG = {
    'dsn': get_secret('SENTRY_DSN_PROD'),
    'environment': 'production',
    'traces_sample_rate': 0.1,  # 10% sampling for production
    'profiles_sample_rate': 0.05,  # 5% profiling
    'send_default_pii': False,  # Compliance requirement
}

# config/environments/development.py  
SENTRY_CONFIG = {
    'dsn': get_secret('SENTRY_DSN_DEV'),
    'environment': 'development', 
    'traces_sample_rate': 1.0,  # 100% sampling for dev
    'profiles_sample_rate': 1.0,  # 100% profiling
    'send_default_pii': True,  # Dev debugging
}
```

### **Feature Flags**

```python
# common/observability/feature_flags.py
class ObservabilityFeatures:
    SENTRY_ENABLED = get_setting('OBSERVABILITY_SENTRY_ENABLED', True)
    DISTRIBUTED_TRACING = get_setting('OBSERVABILITY_TRACING_ENABLED', True)
    PERFORMANCE_MONITORING = get_setting('OBSERVABILITY_PERFORMANCE_ENABLED', True)
    FRONTEND_MONITORING = get_setting('OBSERVABILITY_FRONTEND_ENABLED', True)
    CUSTOM_METRICS = get_setting('OBSERVABILITY_CUSTOM_METRICS_ENABLED', True)
```

## Success Metrics & KPIs

### **Technical Metrics**
- 🎯 **Mean Time to Detection (MTTD)**: <5 minutes for critical errors
- ⚡ **Mean Time to Resolution (MTTR)**: <30 minutes for P0 incidents
- 📊 **Error Rate**: <0.1% for business-critical endpoints
- 🔍 **Observability Coverage**: >95% of code paths instrumented
- 📈 **Alert Accuracy**: <5% false positive rate

### **Business Metrics**
- 💰 **Data Freshness SLA**: 95% of market data <5 minutes stale
- 🕒 **API Availability**: 99.9% uptime during market hours
- 📊 **Pipeline Success Rate**: >99% for daily data processing
- ⚡ **Query Performance**: P95 response time <2 seconds
- 🔔 **Proactive Issue Detection**: >80% of issues detected before user impact

## Cost Estimation

### **Sentry Pricing Tiers**
- **Team Plan**: $26/month for up to 50K errors + 10M performance units
- **Organization Plan**: $80/month for up to 200K errors + 100M performance units
- **Enterprise**: Custom pricing for high-volume applications

**Estimated Monthly Usage:**
- **Error Events**: ~20K/month (based on current error patterns)
- **Transaction Events**: ~2M/month (API calls + Celery tasks)  
- **Performance Monitoring**: ~50M units/month

**Recommended Plan**: **Organization Plan ($80/month)** with room for growth

## Security & Compliance Considerations

### **Data Privacy**
- ✅ **PII Filtering**: No personally identifiable information sent to Sentry
- ✅ **Trading Data Protection**: Symbol and price data classified as non-sensitive
- ✅ **Credential Safety**: API keys and secrets filtered from error reports
- ✅ **Data Residency**: Sentry data hosted in US region for compliance

### **Compliance Requirements**
- 📋 **Audit Trail**: All errors and performance issues logged for compliance review
- 🔒 **Access Control**: Sentry access limited to authorized personnel only  
- 📊 **Retention Policy**: Error data retained for 12 months per regulatory requirements
- 🔍 **Monitoring Integrity**: Real-time alerts for monitoring system failures

## Risk Mitigation

### **High-Impact Risks**

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|---------|------------|
| **Sentry Service Outage** | Low | Medium | Graceful degradation, local logging backup |
| **Performance Overhead** | Medium | Low | Sampling configuration, performance testing |  
| **Alert Fatigue** | High | Medium | Intelligent alerting, proper categorization |
| **Data Leakage** | Low | High | Comprehensive PII filtering, security review |
| **Cost Overrun** | Medium | Low | Usage monitoring, automatic sampling adjustment |

### **Rollback Strategy**
- **Gradual Deployment**: Service-by-service rollout with immediate rollback capability
- **Feature Flags**: Instant disable of Sentry integration without code deployment
- **Fallback Monitoring**: Existing dashboard and logging system remains operational
- **Zero-Downtime**: All observability enhancements designed for zero service disruption

## Next Steps

1. **Immediate Actions (Week 1)**:
   - [ ] Create Sentry organization and project
   - [ ] Generate DSN keys for each environment
   - [ ] Set up initial development environment integration

2. **Team Preparation**:
   - [ ] Schedule Sentry training session for development team
   - [ ] Define on-call rotation and escalation procedures
   - [ ] Create incident response runbooks

3. **Documentation**:
   - [ ] Update deployment guides with Sentry configuration
   - [ ] Create troubleshooting guides for common observability issues
   - [ ] Document custom metric definitions and business logic

---

*This specification provides a comprehensive roadmap for implementing world-class observability in our trading platform, leveraging Sentry's powerful capabilities while building on our existing robust infrastructure.*