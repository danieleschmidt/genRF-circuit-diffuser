# Deployment Documentation

This document provides comprehensive deployment instructions for the GenRF Circuit Diffuser project across different environments.

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/your-org/genrf-circuit-diffuser.git
cd genrf-circuit-diffuser

# Start development environment
docker-compose up -d genrf-dev

# Access dashboard
open http://localhost:3000
```

### Production Deployment

```bash
# Build production image
./scripts/build.sh production --push

# Deploy with docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

## Deployment Options

### 1. Docker Compose (Recommended for Development)

#### Development Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  genrf-dev:
    build:
      context: .
      target: development
    volumes:
      - .:/app
      - genrf-models:/app/models
    ports:
      - "3000:3000"
    environment:
      - GENRF_ENV=development
```

Start development environment:

```bash
docker-compose up -d
```

#### Production Setup

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  genrf-api:
    image: genrf/genrf-circuit-diffuser:production-latest
    restart: unless-stopped
    environment:
      - GENRF_ENV=production
      - DATABASE_URL=postgresql://user:pass@db:5432/genrf
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:14
    restart: unless-stopped
    environment:
      - POSTGRES_DB=genrf
      - POSTGRES_USER=genrf
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - genrf-api

volumes:
  postgres_data:
```

### 2. Kubernetes

#### Namespace Setup

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: genrf
```

#### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genrf-api
  namespace: genrf
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genrf-api
  template:
    metadata:
      labels:
        app: genrf-api
    spec:
      containers:
      - name: genrf-api
        image: genrf/genrf-circuit-diffuser:production-v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: GENRF_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: genrf-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: genrf-api-service
  namespace: genrf
spec:
  selector:
    app: genrf-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: genrf-ingress
  namespace: genrf
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.genrf.example.com
    secretName: genrf-tls
  rules:
  - host: api.genrf.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: genrf-api-service
            port:
              number: 80
```

Deploy to Kubernetes:

```bash
kubectl apply -f k8s/
```

### 3. Cloud Platforms

#### AWS ECS

```json
{
  "family": "genrf-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "genrf-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/genrf:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "GENRF_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:genrf/database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/genrf",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Run

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: genrf-api
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "1000m"
    spec:
      containers:
      - image: gcr.io/your-project/genrf:latest
        ports:
        - containerPort: 8000
        env:
        - name: GENRF_ENV
          value: "production"
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

Deploy:

```bash
gcloud run replace cloudrun.yaml
```

#### Azure Container Instances

```yaml
# aci.yaml
apiVersion: '2019-12-01'
location: eastus
name: genrf-container-group
properties:
  containers:
  - name: genrf-api
    properties:
      image: your-registry.azurecr.io/genrf:latest
      resources:
        requests:
          cpu: 1
          memoryInGb: 2
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: GENRF_ENV
        value: production
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
```

## Configuration

### Environment Variables

#### Core Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GENRF_ENV` | Environment (dev/prod) | development | No |
| `GENRF_PORT` | Application port | 8000 | No |
| `GENRF_LOG_LEVEL` | Log level | INFO | No |

#### Database

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | Database connection | sqlite:///./genrf.db | No |
| `REDIS_URL` | Cache connection | redis://localhost:6379 | No |

#### AI/ML Models

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MODEL_PATH` | Model directory | ./models | No |
| `TORCH_DEVICE` | PyTorch device | cpu | No |
| `MAX_BATCH_SIZE` | Inference batch size | 32 | No |

#### SPICE Integration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SPICE_ENGINE` | SPICE simulator | ngspice | No |
| `SPICE_TIMEOUT` | Simulation timeout | 300 | No |
| `SPICE_MAX_CONCURRENT` | Max parallel sims | 4 | No |

### Configuration Files

#### Application Config

```yaml
# config/production.yaml
app:
  name: "GenRF Circuit Diffuser"
  version: "1.0.0"
  debug: false

database:
  url: "postgresql://user:pass@localhost:5432/genrf"
  pool_size: 10
  max_overflow: 20

redis:
  url: "redis://localhost:6379"
  max_connections: 10

spice:
  engine: "ngspice"
  timeout: 300
  max_concurrent: 4

models:
  path: "/app/models"
  cache_size: "1GB"
  
logging:
  level: "INFO"
  format: "json"
  
security:
  secret_key: "${SECRET_KEY}"
  cors_origins:
    - "https://app.genrf.com"
    - "https://dashboard.genrf.com"
```

#### Nginx Configuration

```nginx
# nginx.conf
upstream genrf_api {
    server genrf-api:8000;
}

server {
    listen 80;
    server_name api.genrf.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.genrf.example.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    client_max_body_size 100M;
    
    location / {
        proxy_pass http://genrf_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://genrf_api/health;
    }
}
```

## Monitoring and Observability

### Health Checks

The application provides several health check endpoints:

- `/health` - Basic health check
- `/ready` - Readiness check
- `/metrics` - Prometheus metrics

### Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'genrf-api'
    static_configs:
      - targets: ['genrf-api:9090']
    metrics_path: /metrics
    scrape_interval: 30s
```

### Grafana Dashboard

Import the pre-built dashboard from `monitoring/grafana-dashboard.json`.

Key metrics:
- Request rate and latency
- Circuit generation success rate
- SPICE simulation performance
- Model inference time
- Resource utilization

### Logging

#### Structured Logging

```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "level": "INFO",
  "service": "genrf-api",
  "trace_id": "abc123",
  "message": "Circuit generation completed",
  "circuit_type": "LNA",
  "generation_time": 2.5,
  "success": true
}
```

#### Log Aggregation

**ELK Stack:**

```yaml
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "genrf-api" {
    json {
      source => "message"
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "genrf-logs-%{+YYYY.MM.dd}"
  }
}
```

## Security

### SSL/TLS Configuration

```bash
# Generate self-signed certificate for testing
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem \
  -out ssl/cert.pem \
  -subj "/CN=api.genrf.example.com"
```

### Secrets Management

#### Kubernetes Secrets

```bash
# Create database secret
kubectl create secret generic genrf-secrets \
  --from-literal=database-url="postgresql://user:pass@db:5432/genrf" \
  --namespace=genrf
```

#### Docker Secrets

```yaml
# docker-compose.yml
secrets:
  database_url:
    file: ./secrets/database_url.txt

services:
  genrf-api:
    secrets:
      - database_url
    environment:
      - DATABASE_URL_FILE=/run/secrets/database_url
```

### Network Security

```yaml
# Security groups / firewall rules
- port: 80/443 (HTTP/HTTPS)
  source: 0.0.0.0/0
- port: 8000 (API)
  source: internal network only
- port: 5432 (PostgreSQL)
  source: application subnet only
```

## Scaling

### Horizontal Scaling

#### Kubernetes HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: genrf-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genrf-api
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

### Vertical Scaling

```yaml
# Increase resource limits
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup
pg_dump -h localhost -U genrf -d genrf > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup with cron
0 2 * * * /usr/local/bin/pg_dump -h db -U genrf -d genrf | gzip > /backups/genrf_$(date +\%Y\%m\%d).sql.gz
```

### Model Backup

```bash
# Sync models to cloud storage
aws s3 sync /app/models s3://genrf-models-backup/$(date +%Y%m%d)/
```

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs genrf-api

# Common issues:
# - Missing environment variables
# - Database connection failure
# - Model files not found
```

#### High Memory Usage

```bash
# Monitor memory usage
docker stats genrf-api

# Solutions:
# - Reduce MAX_BATCH_SIZE
# - Increase memory limits
# - Enable model quantization
```

#### SPICE Simulation Failures

```bash
# Check SPICE engine
docker exec genrf-api ngspice --version

# Common issues:
# - NgSpice not installed
# - Netlist syntax errors
# - Simulation timeout
```

### Debug Mode

```bash
# Enable debug logging
docker run -e GENRF_LOG_LEVEL=DEBUG genrf-api

# Enable profiling
docker run -e ENABLE_PROFILING=true genrf-api
```

## Performance Optimization

### Container Optimization

```dockerfile
# Multi-stage build to reduce size
FROM python:3.11-slim as base
# ... install only necessary packages

FROM base as production
# ... copy only production files
```

### Resource Tuning

```yaml
# Optimize for your workload
resources:
  requests:
    memory: "2Gi"    # Based on model size
    cpu: "1000m"     # Based on inference needs
  limits:
    memory: "4Gi"    # Allow burst capacity
    cpu: "2000m"     # Prevent noisy neighbors
```

### Caching Strategy

```yaml
# Redis for caching
environment:
  - REDIS_URL=redis://redis:6379
  - CACHE_TTL=3600  # 1 hour cache
```

For more deployment scenarios and advanced configurations, see:
- [Production Deployment Guide](production-deployment.md)
- [Kubernetes Best Practices](kubernetes-best-practices.md)
- [Security Hardening Guide](security-hardening.md)