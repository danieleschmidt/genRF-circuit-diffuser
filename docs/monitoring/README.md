# Monitoring and Observability

This document describes the comprehensive monitoring and observability setup for the GenRF Circuit Diffuser project.

## Overview

Our monitoring stack provides full observability across all components of the GenRF system:

- **Metrics Collection**: Prometheus
- **Visualization**: Grafana 
- **Alerting**: Alertmanager
- **Distributed Tracing**: Jaeger
- **Log Aggregation**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Container Monitoring**: cAdvisor
- **System Monitoring**: Node Exporter

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GenRF API     │───▶│   Prometheus    │───▶│    Grafana      │
│  (Metrics)      │    │  (Collection)   │    │ (Visualization) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Alertmanager   │───▶│  Notifications  │
                       │   (Alerting)    │    │ (Email, Slack)  │
                       └─────────────────┘    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│     Jaeger      │───▶│    Tracing      │
│    (Traces)     │    │   (Tracing)     │    │     Analysis    │
└─────────────────┘    └─────────────────┘    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│    Logstash     │───▶│     Kibana      │
│     (Logs)      │    │ (Log Processing)│    │ (Log Analysis)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Elasticsearch   │
                       │ (Log Storage)   │
                       └─────────────────┘
```

## Quick Start

### 1. Start Monitoring Stack

```bash
# Start all monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Verify services are running
docker-compose -f docker-compose.monitoring.yml ps
```

### 2. Access Dashboards

| Service | URL | Default Credentials |
|---------|-----|-------------------|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | None |
| Alertmanager | http://localhost:9093 | None |
| Jaeger | http://localhost:16686 | None |
| Kibana | http://localhost:5601 | None |

### 3. Import Grafana Dashboards

```bash
# Import pre-built dashboards
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana/dashboards/genrf-overview.json
```

## Metrics

### Application Metrics

The GenRF API exposes metrics at `/metrics` endpoint:

#### Circuit Generation Metrics

```prometheus
# Circuit generation requests
genrf_circuit_generation_requests_total{circuit_type, status}

# Circuit generation duration
genrf_circuit_generation_duration_seconds{circuit_type}

# Circuit quality scores
genrf_circuit_quality_score{circuit_type}

# Specification violations
genrf_spec_violations_total{circuit_type, violation_type}
```

#### SPICE Simulation Metrics

```prometheus
# SPICE simulation requests
genrf_spice_simulation_requests_total{engine, status}

# SPICE simulation duration
genrf_spice_simulation_duration_seconds{engine}

# SPICE simulation queue length
genrf_spice_simulation_queue_length

# SPICE simulation timeouts
genrf_spice_simulation_timeouts_total{engine}
```

#### ML Model Metrics

```prometheus
# Model inference requests
genrf_model_inference_requests_total{model_type, status}

# Model inference duration
genrf_model_inference_duration_seconds{model_type}

# Model loading status
genrf_model_loaded{model_name}

# Model memory usage
genrf_model_memory_bytes{model_name}
```

#### HTTP Metrics

```prometheus
# HTTP requests
genrf_http_requests_total{method, endpoint, status}

# HTTP request duration
genrf_http_request_duration_seconds{method, endpoint}

# Active connections
genrf_http_connections_active
```

### System Metrics

#### Node Exporter (System)

- CPU usage, load average
- Memory usage and statistics
- Disk I/O and space usage
- Network statistics
- Process statistics

#### cAdvisor (Containers)

- Container CPU and memory usage
- Container network and disk I/O
- Container filesystem usage

#### Redis Metrics

- Connection count
- Memory usage
- Command statistics
- Keyspace statistics

## Custom Dashboards

### GenRF Overview Dashboard

Main dashboard showing:
- Circuit generation rate and success rate
- Response time percentiles
- Error rate trends
- System resource utilization

### SPICE Simulation Dashboard

SPICE-specific metrics:
- Simulation queue depth
- Simulation success/failure rates
- Engine performance comparison
- Simulation duration distribution

### ML Model Performance Dashboard

Model-specific metrics:
- Inference latency trends
- Model accuracy metrics
- Resource utilization per model
- Batch processing efficiency

### Circuit Quality Dashboard

Quality assurance metrics:
- Specification compliance rates
- Quality score distributions
- Design space coverage
- Convergence statistics

## Alerting

### Alert Categories

#### Critical Alerts
- API service down
- SPICE engine unavailable
- Model loading failures
- Database connectivity issues

#### Warning Alerts
- High error rates (>5%)
- High latency (>30s 95th percentile)
- Resource utilization (>80%)
- Circuit quality degradation

#### Info Alerts
- Deployment notifications
- Configuration changes
- Scheduled maintenance

### Alert Routing

```yaml
# Example alert routing
route:
  routes:
    - match:
        severity: critical
      receiver: 'oncall-team'
      repeat_interval: 1h
    
    - match:
        service: spice-simulator
      receiver: 'simulation-team'
      
    - match:
        service: ml-inference
      receiver: 'ml-team'
```

### Notification Channels

#### Email
- Individual notifications
- Team distribution lists
- Escalation chains

#### Slack
- Channel-specific notifications
- Direct messages for critical alerts
- Alert aggregation

#### PagerDuty
- On-call rotation
- Escalation policies
- Incident management

### Alert Suppression

```yaml
# Suppress warnings during critical outages
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['service']
```

## Distributed Tracing

### Jaeger Configuration

Traces are automatically collected for:
- HTTP requests
- Circuit generation pipelines
- SPICE simulations
- ML model inference
- Database operations

### Trace Analysis

#### Performance Analysis
- Request flow visualization
- Bottleneck identification
- Dependency mapping
- Error propagation

#### Debug Workflows
- Failed request investigation
- Performance regression analysis
- Service dependency analysis

### Custom Instrumentation

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("circuit_generation")
def generate_circuit(spec):
    with tracer.start_as_current_span("validate_spec"):
        validate_specification(spec)
    
    with tracer.start_as_current_span("model_inference"):
        circuit = model.generate(spec)
    
    return circuit
```

## Log Management

### Log Aggregation Pipeline

```
Application Logs → Logstash → Elasticsearch → Kibana
```

### Log Structure

```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "level": "INFO",
  "service": "genrf-api",
  "trace_id": "abc123",
  "span_id": "def456",
  "message": "Circuit generation completed",
  "circuit_type": "LNA",
  "generation_time": 2.5,
  "success": true,
  "user_id": "user123"
}
```

### Log Categories

#### Application Logs
- Request/response logging
- Circuit generation events
- Error conditions
- Performance metrics

#### Security Logs
- Authentication events
- Authorization failures
- API key usage
- Suspicious activities

#### Audit Logs
- Configuration changes
- User actions
- System modifications
- Compliance events

### Log Retention

| Log Type | Retention Period | Storage |
|----------|------------------|---------|
| Application | 30 days | Hot storage |
| Error | 90 days | Warm storage |
| Security | 1 year | Cold storage |
| Audit | 7 years | Archive |

## Performance Monitoring

### SLA Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| API Availability | 99.9% | Monthly uptime |
| Response Time | <5s (95th percentile) | HTTP requests |
| Circuit Generation | <30s | End-to-end pipeline |
| SPICE Simulation | <60s | Individual simulation |

### Performance Benchmarks

#### Circuit Generation Performance

```prometheus
# Target: <30 seconds for 95% of requests
histogram_quantile(0.95, 
  rate(genrf_circuit_generation_duration_seconds_bucket[5m])
) < 30
```

#### API Performance

```prometheus
# Target: <5 seconds for 95% of requests
histogram_quantile(0.95,
  rate(genrf_http_request_duration_seconds_bucket[5m])
) < 5
```

### Capacity Planning

#### Resource Utilization Trends
- CPU usage patterns
- Memory consumption growth
- Disk space usage
- Network bandwidth

#### Scaling Indicators
- Request rate trends
- Queue depth monitoring
- Response time degradation
- Resource saturation points

## Health Checks

### Application Health

```http
GET /health
{
  "status": "healthy",
  "checks": {
    "database": "ok",
    "redis": "ok", 
    "spice_engine": "ok",
    "models": "ok"
  },
  "uptime": "72h45m",
  "version": "1.0.0"
}
```

### Readiness Checks

```http
GET /ready
{
  "ready": true,
  "checks": {
    "models_loaded": true,
    "database_migrations": true,
    "cache_warmed": true
  }
}
```

## Runbooks

### Common Issues

#### High Circuit Generation Latency
1. Check SPICE simulation queue
2. Verify model inference performance
3. Check resource utilization
4. Scale simulation workers if needed

#### SPICE Engine Failures
1. Check SPICE engine status
2. Verify netlist syntax
3. Check simulation timeouts
4. Restart SPICE engine if needed

#### Model Inference Errors
1. Check model loading status
2. Verify input data format
3. Check GPU availability
4. Reload models if corrupted

## Security Monitoring

### Security Metrics

```prometheus
# Failed authentication attempts
genrf_auth_failures_total{reason}

# API rate limiting
genrf_rate_limit_violations_total{endpoint}

# Suspicious activities
genrf_security_events_total{event_type}
```

### Security Alerts

- Multiple failed login attempts
- API rate limit violations
- Unusual access patterns
- Privileged operation failures

## Troubleshooting

### Common Monitoring Issues

#### Metrics Not Appearing
```bash
# Check if application is exposing metrics
curl http://localhost:8000/metrics

# Verify Prometheus configuration
docker exec genrf-prometheus promtool check config /etc/prometheus/prometheus.yml

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

#### Grafana Dashboard Issues
```bash
# Check Grafana logs
docker logs genrf-grafana

# Verify data source connection
curl -u admin:admin http://localhost:3000/api/datasources
```

#### Alert Not Firing
```bash
# Check alert rule syntax
docker exec genrf-prometheus promtool check rules /etc/prometheus/alerts.yml

# Verify Alertmanager routing
curl http://localhost:9093/api/v1/status
```

### Debug Commands

```bash
# View all monitoring services
docker-compose -f docker-compose.monitoring.yml ps

# Check service logs
docker logs genrf-prometheus
docker logs genrf-grafana
docker logs genrf-alertmanager

# Restart monitoring stack
docker-compose -f docker-compose.monitoring.yml restart
```

For more detailed monitoring configurations, see:
- [Grafana Dashboard Configurations](grafana-dashboards.md)
- [Custom Metrics Implementation](custom-metrics.md)
- [Alert Runbook](alert-runbook.md)