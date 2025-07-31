# Observability and Monitoring Guide

This document outlines monitoring, logging, and observability practices for the genRF project.

## Overview

GenRF requires comprehensive observability to ensure:
- Circuit generation performance and quality
- SPICE simulation health and efficiency
- Resource utilization and scaling behavior
- User experience and error patterns

## Monitoring Stack

### 1. Application Metrics

**Recommended Tools**: Prometheus + Grafana

**Key Metrics to Track**:

```python
# Example metrics in genrf/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# Generation metrics
circuit_generation_total = Counter(
    'genrf_circuits_generated_total',
    'Total circuits generated',
    ['circuit_type', 'technology', 'status']
)

generation_duration = Histogram(
    'genrf_generation_duration_seconds',
    'Time spent generating circuits',
    ['circuit_type', 'optimization_steps'],
    buckets=[1, 5, 10, 30, 60, 300, 900]  # 1s to 15min
)

# SPICE simulation metrics
spice_simulation_total = Counter(
    'genrf_spice_simulations_total',
    'Total SPICE simulations run',
    ['engine', 'circuit_type', 'status']
)

spice_convergence_issues = Counter(
    'genrf_spice_convergence_failures_total',
    'SPICE convergence failures',
    ['circuit_type', 'failure_type']
)

# Resource metrics
gpu_memory_usage = Gauge(
    'genrf_gpu_memory_bytes',
    'GPU memory usage',
    ['gpu_id']
)

model_inference_duration = Histogram(
    'genrf_model_inference_seconds',
    'Neural network inference time',
    ['model_type', 'batch_size']
)

# Quality metrics
circuit_performance = Histogram(
    'genrf_circuit_performance',
    'Generated circuit performance metrics',
    ['metric_type', 'circuit_type', 'technology'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
)

# Business metrics
user_satisfaction = Histogram(
    'genrf_user_satisfaction_score',
    'User satisfaction ratings',
    ['feature'],
    buckets=[1, 2, 3, 4, 5]
)
```

### 2. Logging Strategy

**Tool**: Structured logging with Python `structlog`

```python
# Example logging configuration
import structlog
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(colors=False),
            "foreign_pre_chain": [
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.ExtraAdder(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
            ],
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "genrf.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
        },
    },
    "loggers": {
        "genrf": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
        "genrf.security": {
            "handlers": ["file"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}
```

**Log Levels and Content**:
- **ERROR**: Circuit generation failures, SPICE crashes, security violations
- **WARN**: Performance degradation, convergence issues, resource constraints
- **INFO**: Generation requests, completion status, user actions
- **DEBUG**: Detailed algorithm steps, parameter values (non-sensitive)

### 3. Distributed Tracing

**Tool**: OpenTelemetry + Jaeger

```python
# Example tracing setup
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_tracing():
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger-agent",
        agent_port=6831,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    return tracer

# Usage in circuit generation
@trace.get_tracer(__name__).start_as_current_span("generate_circuit")
def generate_circuit(spec):
    span = trace.get_current_span()
    span.set_attribute("circuit.type", spec.circuit_type)
    span.set_attribute("circuit.frequency", spec.frequency)
    
    with trace.get_tracer(__name__).start_as_current_span("topology_generation"):
        topology = generate_topology(spec)
        
    with trace.get_tracer(__name__).start_as_current_span("parameter_optimization"):
        parameters = optimize_parameters(topology, spec)
        
    with trace.get_tracer(__name__).start_as_current_span("spice_validation"):
        results = validate_with_spice(topology, parameters)
        
    return create_circuit(topology, parameters, results)
```

## Grafana Dashboards

### 1. System Health Dashboard

**Panels**:
- Circuit generation rate (circuits/hour)
- Success rate percentage
- Average generation time
- SPICE simulation success rate
- GPU/CPU utilization
- Memory usage trends

### 2. Performance Dashboard

**Panels**:
- Generation time percentiles (P50, P95, P99)
- Model inference latency
- SPICE convergence rate by circuit type
- Optimization iteration counts
- Queue depths and processing times

### 3. Business Metrics Dashboard

**Panels**:
- Active users and sessions
- Popular circuit types and frequencies
- Export format usage
- User satisfaction scores
- Feature adoption rates

### 4. Error Analysis Dashboard

**Panels**:
- Error rates by component
- Top error messages
- Failure patterns by technology/frequency
- Recovery time metrics
- Alert escalation status

## Alerting Rules

### Critical Alerts (PagerDuty/On-call)

```yaml
# Prometheus alerting rules
groups:
  - name: genrf.critical
    rules:
      - alert: CircuitGenerationDown
        expr: rate(genrf_circuits_generated_total[5m]) == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Circuit generation stopped"
          description: "No circuits generated in the last 5 minutes"
          
      - alert: HighErrorRate
        expr: |
          (
            rate(genrf_circuits_generated_total{status="failed"}[5m]) /
            rate(genrf_circuits_generated_total[5m])
          ) > 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High circuit generation error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"
          
      - alert: SPICESimulationFailures
        expr: rate(genrf_spice_convergence_failures_total[5m]) > 10
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High SPICE simulation failure rate"
          description: "{{ $value }} SPICE failures per second"
```

### Warning Alerts (Slack/Email)

```yaml
  - name: genrf.warning
    rules:  
      - alert: SlowGenerationTime
        expr: |
          histogram_quantile(0.95, 
            rate(genrf_generation_duration_seconds_bucket[10m])
          ) > 300
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Circuit generation is slow"
          description: "95th percentile generation time is {{ $value }}s"
          
      - alert: HighGPUMemoryUsage
        expr: genrf_gpu_memory_bytes / (8 * 1024^3) > 0.9  # >90% of 8GB
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High GPU memory usage"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"
```

## Performance Profiling

### 1. Application Profiling

```python
# Built-in profiling hooks
import cProfile
import pstats
from functools import wraps

def profile_generation(func):
    """Decorator to profile circuit generation functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            
            # Save profile data
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.dump_stats(f'profile_{func.__name__}_{int(time.time())}.prof')
            
        return result
    return wrapper

# Usage
@profile_generation
def generate_lna_circuit(spec):
    # Circuit generation logic
    pass
```

### 2. Memory Profiling

```python
# Memory usage tracking
import tracemalloc
import psutil
import os

def track_memory_usage():
    """Context manager for tracking memory usage."""
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss
    
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        end_memory = process.memory_info().rss
        tracemalloc.stop()
        
        logger.info(
            "Memory usage",
            current_traced=current,
            peak_traced=peak,
            rss_start=start_memory,
            rss_end=end_memory,
            rss_delta=end_memory - start_memory
        )
```

## Log Analysis

### 1. ELK Stack Integration

**Elasticsearch mapping for genRF logs**:
```json
{
  "mappings": {
    "properties": {
      "timestamp": {"type": "date"},
      "level": {"type": "keyword"},
      "logger": {"type": "keyword"},
      "message": {"type": "text"},
      "circuit_type": {"type": "keyword"},
      "technology": {"type": "keyword"},
      "user_id": {"type": "keyword"},
      "session_id": {"type": "keyword"},
      "generation_time": {"type": "float"},
      "spice_engine": {"type": "keyword"}
    }
  }
}
```

### 2. Common Log Queries

```
# Find slow generations
level:INFO AND generation_time:>300

# SPICE convergence issues  
level:ERROR AND "convergence failure"

# User error patterns
level:ERROR AND user_id:* | terms field:user_id size:10

# Technology-specific issues
circuit_type:LNA AND technology:tsmc65 AND level:ERROR
```

## Health Checks

### 1. Application Health

```python
# Health check endpoint
from flask import Flask, jsonify
import time

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Comprehensive health check."""
    health_status = {
        "timestamp": time.time(),
        "status": "healthy",
        "checks": {}
    }
    
    # Check database connection
    try:
        # Mock database check
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {e}"
        health_status["status"] = "unhealthy"
    
    # Check SPICE engine
    try:
        # Mock SPICE check
        health_status["checks"]["spice"] = "healthy"
    except Exception as e:
        health_status["checks"]["spice"] = f"unhealthy: {e}"
        health_status["status"] = "unhealthy"
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            health_status["checks"]["gpu"] = "healthy"
        else:
            health_status["checks"]["gpu"] = "no_gpu"
    except Exception as e:
        health_status["checks"]["gpu"] = f"unhealthy: {e}"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    # Return Prometheus formatted metrics
    pass
```

### 2. Infrastructure Health

```bash
#!/bin/bash
# health_check.sh - Infrastructure health check script

# Check disk space
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    echo "CRITICAL: Disk usage at ${DISK_USAGE}%"
    exit 2
fi

# Check memory usage
MEM_USAGE=$(free | awk 'FNR==2{printf "%.0f", $3/($3+$4)*100}')
if [ $MEM_USAGE -gt 90 ]; then
    echo "WARNING: Memory usage at ${MEM_USAGE}%"
    exit 1
fi

# Check if services are running
if ! pgrep -f "genrf" > /dev/null; then
    echo "CRITICAL: GenRF service not running"
    exit 2
fi

echo "OK: All health checks passed"
exit 0
```

## Capacity Planning

### 1. Resource Usage Trends

Track these metrics for capacity planning:
- Circuit generation requests per hour/day
- Average generation time by circuit complexity
- Memory usage per concurrent generation
- GPU utilization patterns
- Storage growth rate for models and outputs

### 2. Scaling Triggers

```yaml
# Example Kubernetes HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: genrf-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genrf-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: pending_generations
      target:
        type: AverageValue
        averageValue: "5"
```

This observability setup provides comprehensive monitoring and insights into the genRF system's performance, health, and user experience.