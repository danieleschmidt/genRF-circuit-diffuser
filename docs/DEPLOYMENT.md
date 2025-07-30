# Deployment Guide

This guide covers deployment strategies and operational considerations for GenRF Circuit Diffuser.

## Deployment Options

### 1. Local Development

For development and experimentation:

```bash
# Install from source
git clone https://github.com/yourusername/genRF-circuit-diffuser.git
cd genRF-circuit-diffuser
pip install -e ".[dev,spice]"

# Run CLI
genrf generate --spec examples/lna_spec.yaml
```

### 2. Docker Deployment

#### Production Container

```bash
# Build production image
docker build --target production -t genrf:latest .

# Run circuit generation
docker run -v $(pwd)/outputs:/app/outputs genrf:latest \
  generate --spec /app/configs/lna_spec.yaml --output /app/outputs
```

#### Docker Compose

```bash
# Start complete stack
docker-compose up -d

# Access dashboard at http://localhost:3000
# Access Jupyter at http://localhost:8888
```

### 3. Cloud Deployment

#### AWS ECS

```yaml
# ecs-task-definition.json
{
  "family": "genrf-circuit-diffuser",
  "taskRoleArn": "arn:aws:iam::123456789012:role/genrf-task-role",
  "executionRoleArn": "arn:aws:iam::123456789012:role/genrf-execution-role",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "genrf",
      "image": "your-registry/genrf:latest",
      "essential": true,
      "memory": 4096,
      "cpu": 2048,
      "environment": [
        {"name": "GENRF_ENV", "value": "production"}
      ],
      "mountPoints": [
        {
          "sourceVolume": "models",
          "containerPath": "/app/models",
          "readOnly": true
        }
      ]
    }
  ],
  "volumes": [
    {
      "name": "models",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-1234567890abcdef0"
      }
    }
  ]
}
```

#### Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genrf-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genrf
  template:
    metadata:
      labels:
        app: genrf
    spec:
      containers:
      - name: genrf
        image: your-registry/genrf:latest
        ports:
        - containerPort: 3000
        env:
        - name: GENRF_ENV
          value: "production"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: genrf-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: genrf-service
spec:
  selector:
    app: genrf
  ports:
  - protocol: TCP
    port: 80
    targetPort: 3000
  type: LoadBalancer
```

## Configuration Management

### Environment Variables

```bash
# Core configuration
export GENRF_ENV=production
export GENRF_LOG_LEVEL=INFO
export GENRF_MODELS_PATH=/app/models

# SPICE configuration
export SPICE_ENGINE=ngspice
export SPICE_MODELS_PATH=/app/spice_models

# Performance tuning
export GENRF_MAX_WORKERS=4
export GENRF_BATCH_SIZE=32
export GENRF_GPU_MEMORY_FRACTION=0.8

# Security
export GENRF_API_KEY_FILE=/secrets/api_key
export GENRF_TLS_CERT_FILE=/certs/tls.crt
export GENRF_TLS_KEY_FILE=/certs/tls.key
```

### Configuration Files

Create `config/production.yaml`:

```yaml
# Production configuration
app:
  log_level: INFO
  max_workers: 4
  enable_metrics: true

models:
  path: /app/models
  cache_size: 1000
  preload_models: true

spice:
  engine: ngspice
  models_path: /app/spice_models
  simulation_timeout: 300
  max_concurrent_sims: 8

dashboard:
  host: 0.0.0.0
  port: 3000
  enable_authentication: true

security:
  api_key_required: true
  rate_limit_per_minute: 100
  cors_origins: []

monitoring:
  enable_prometheus: true
  metrics_port: 9090
  health_check_interval: 30
```

## Security Considerations

### Network Security

- Use HTTPS/TLS for all external communications
- Implement API authentication and rate limiting
- Restrict network access with firewalls/security groups
- Use VPN for administrative access

### Container Security

```dockerfile
# Security-hardened Dockerfile additions
FROM python:3.10-slim AS production

# Create non-root user
RUN groupadd -r genrf && useradd -r -g genrf genrf

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set secure permissions
COPY --chown=genrf:genrf --chmod=755 . /app

# Run as non-root
USER genrf

# Security labels
LABEL security.scan="completed" \
      security.cve.check="passed"
```

### Data Protection

- Encrypt model files at rest
- Use secrets management for API keys
- Implement audit logging
- Regular security vulnerability scanning

## Monitoring and Observability

### Health Checks

```python
# Health check endpoint
@app.route('/health')
def health_check():
    checks = {
        'database': check_database_connection(),
        'spice_engine': check_spice_availability(),
        'models': check_model_loading(),
        'disk_space': check_disk_space()
    }
    
    status = 'healthy' if all(checks.values()) else 'unhealthy'
    return {'status': status, 'checks': checks}
```

### Metrics Collection

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

circuit_generation_counter = Counter(
    'genrf_circuits_generated_total',
    'Total circuits generated',
    ['circuit_type', 'status']
)

generation_time_histogram = Histogram(
    'genrf_generation_duration_seconds',
    'Circuit generation duration',
    ['circuit_type']
)

model_memory_gauge = Gauge(
    'genrf_model_memory_bytes',
    'Model memory usage'
)
```

### Logging Configuration

```yaml
# logging.yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  json:
    format: '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: json
    filename: /var/log/genrf/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  genrf:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: WARNING
  handlers: [console]
```

## Performance Optimization

### Resource Requirements

**Minimum Requirements**:
- CPU: 2 cores, 2.4 GHz
- Memory: 4 GB RAM
- Storage: 10 GB SSD
- Network: 100 Mbps

**Recommended for Production**:
- CPU: 8 cores, 3.0 GHz
- Memory: 16 GB RAM
- Storage: 100 GB NVMe SSD
- Network: 1 Gbps
- GPU: NVIDIA GPU with 8GB VRAM (optional)

### Scaling Strategies

#### Horizontal Scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: genrf-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genrf-deployment
  minReplicas: 2
  maxReplicas: 10
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
```

#### Vertical Scaling

```bash
# Resource limits for single instance
docker run --cpus="4.0" --memory="8g" \
  -v models:/app/models \
  genrf:latest
```

### Caching Strategy

```python
# Model caching
from functools import lru_cache
import redis

# Memory cache for small models
@lru_cache(maxsize=5)
def load_model(model_path):
    return torch.load(model_path)

# Redis cache for circuit results
redis_client = redis.Redis(host='cache-server')

def get_cached_circuit(spec_hash):
    return redis_client.get(f"circuit:{spec_hash}")

def cache_circuit(spec_hash, circuit_data):
    redis_client.setex(f"circuit:{spec_hash}", 3600, circuit_data)
```

## Backup and Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh - Backup script

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup models
tar -czf "$BACKUP_DIR/models.tar.gz" /app/models/

# Backup generated circuits
tar -czf "$BACKUP_DIR/outputs.tar.gz" /app/outputs/

# Backup configuration
cp -r /app/config "$BACKUP_DIR/"

# Upload to S3 (if using AWS)
aws s3 sync "$BACKUP_DIR" s3://genrf-backups/
```

### Disaster Recovery

1. **Recovery Time Objective (RTO)**: 30 minutes
2. **Recovery Point Objective (RPO)**: 1 hour
3. **Backup retention**: 30 days

### High Availability

```yaml
# Multi-region deployment
regions:
  primary: us-east-1
  secondary: us-west-2

load_balancer:
  health_check_path: /health
  healthy_threshold: 2
  unhealthy_threshold: 3
  timeout: 5

failover:
  automatic: true
  health_check_grace_period: 300
```

## Maintenance

### Regular Tasks

- **Daily**: Monitor system health and performance
- **Weekly**: Review logs and error rates
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Performance benchmarking and optimization

### Update Procedure

```bash
# Rolling update procedure
1. Deploy to staging environment
2. Run full test suite
3. Backup production data
4. Deploy to production with blue-green strategy
5. Verify functionality
6. Route traffic to new version
```

### Troubleshooting

Common issues and solutions:

1. **High memory usage**: Scale vertically or implement model sharding
2. **SPICE simulation timeouts**: Increase timeout or optimize netlists
3. **Slow circuit generation**: Check model loading and GPU utilization
4. **Network connectivity**: Verify firewall rules and DNS resolution

For detailed troubleshooting, see the monitoring dashboards and log analysis tools.