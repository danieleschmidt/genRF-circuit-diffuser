# ADR-0001: ML Framework Selection (PyTorch vs TensorFlow)

## Status
Accepted

## Context

The GenRF Circuit Diffuser requires a machine learning framework for implementing:
- Cycle-GAN for circuit topology generation
- Diffusion models for parameter optimization
- Bayesian optimization for multi-objective design

Key requirements:
- Research-friendly for experimenting with novel architectures
- Strong GPU acceleration support
- Good ecosystem for generative models
- Production deployment capabilities
- Active community and long-term support

## Decision

We will use **PyTorch** as the primary ML framework for GenRF Circuit Diffuser.

## Consequences

### Positive
- **Research Flexibility**: Dynamic computational graphs enable easy experimentation with novel architectures
- **Generative Model Ecosystem**: Strong support for GANs and diffusion models through libraries like `torch-audio`, `transformers`, and custom implementations
- **GPU Performance**: Excellent CUDA integration and optimization
- **Python-First Design**: Natural integration with Python-based scientific computing stack
- **Active Community**: Large community focusing on research and cutting-edge techniques
- **Industry Adoption**: Widespread use in both research and production (Meta, Tesla, OpenAI)

### Negative
- **Model Serving**: Requires additional tools (TorchServe, ONNX) for production deployment
- **Mobile Deployment**: Less mature than TensorFlow Lite for edge deployment
- **Graph Optimization**: Requires TorchScript for graph-level optimizations

### Neutral
- **Learning Curve**: Team already has PyTorch experience
- **Ecosystem**: Both frameworks have mature ecosystems

## Alternatives Considered

### TensorFlow 2.x
**Pros**:
- Mature production ecosystem (TF Serving, TF Lite)
- Strong graph optimization for inference
- Robust distributed training support
- Google's long-term commitment

**Cons**:
- Eager execution still feels less natural than PyTorch
- Less popular in generative modeling research community
- More complex for research and experimentation
- Keras abstraction sometimes limits flexibility

**Verdict**: While TensorFlow has excellent production tooling, PyTorch's advantages for research and generative modeling outweigh the production considerations for our use case.

### JAX
**Pros**:
- Functional programming paradigm
- Excellent performance with XLA compilation
- Growing research adoption
- Built-in transformations (grad, jit, vmap)

**Cons**:
- Smaller ecosystem, especially for generative models
- Steeper learning curve
- Less mature production tooling
- Limited pre-trained model availability

**Verdict**: Too early in adoption for a production system, though we may consider it for research experiments.

### Other Frameworks
- **MXNet**: Declining community support
- **PaddlePaddle**: Limited international ecosystem
- **OneFlow**: Too new and limited adoption

## Implementation Plan

### Phase 1: Core Infrastructure
1. Set up PyTorch environment with CUDA support
2. Implement base model architectures (GAN, Diffusion)
3. Create model training and inference pipelines
4. Establish model serialization and versioning

### Phase 2: Production Integration
1. Implement TorchServe for model serving
2. Set up ONNX export for cross-platform inference
3. Create model monitoring and performance tracking
4. Implement model A/B testing infrastructure

### Phase 3: Optimization
1. Optimize models with TorchScript
2. Implement model quantization for inference
3. Set up distributed training for large models
4. Create automated model retraining pipelines

## Technical Specifications

### Minimum PyTorch Version
- **PyTorch**: 1.12.0+
- **TorchVision**: 0.13.0+
- **CUDA**: 11.8+ for GPU support

### Key Libraries
- **Transformers**: For attention mechanisms in diffusion models
- **TorchServe**: Model serving in production
- **PyTorch Lightning**: Training loop abstraction and multi-GPU support
- **Weights & Biases**: Experiment tracking and model versioning

### Model Architecture Standards
```python
# Base model interface
class BaseCircuitModel(nn.Module):
    """Base class for all circuit generation models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def generate(self, spec: DesignSpec) -> torch.Tensor:
        """Generate circuit based on specification."""
        raise NotImplementedError
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'version': self.__class__.__version__
        }, path)
    
    @classmethod
    def load_checkpoint(cls, path: str) -> 'BaseCircuitModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Model Inference | < 5 seconds | Single circuit generation |
| Training Time | < 24 hours | Full model training on single GPU |
| Memory Usage | < 8GB GPU | For largest models during inference |
| Model Size | < 500MB | Serialized model checkpoint |

## Risk Mitigation

### Framework Migration Risk
- **Risk**: PyTorch becomes obsolete or unsupported
- **Mitigation**: ONNX export provides framework portability
- **Monitoring**: Track PyTorch development and community health

### Performance Risk
- **Risk**: PyTorch performance insufficient for production
- **Mitigation**: TorchScript optimization and ONNX runtime
- **Fallback**: Model architecture optimization and hardware scaling

### Ecosystem Risk
- **Risk**: Required libraries become unavailable
- **Mitigation**: Pin dependency versions and maintain mirrors
- **Contingency**: Implement custom versions of critical components

## Success Metrics

### Technical Metrics
- Model training convergence within expected timeframes
- Inference latency meeting performance targets
- Memory usage within hardware constraints
- Model accuracy meeting quality benchmarks

### Team Metrics
- Developer productivity and satisfaction
- Time to implement new model architectures
- Debugging and troubleshooting efficiency
- Community resource availability

## Timeline

- **Week 1-2**: Environment setup and basic model implementation
- **Week 3-4**: Training pipeline and first model versions
- **Week 5-6**: Production serving infrastructure
- **Week 7-8**: Performance optimization and monitoring
- **Week 9-12**: Advanced features and production deployment

## References

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [PyTorch Production Deployment Guide](https://pytorch.org/tutorials/intermediate/torchserve_tutorial.html)
- [ONNX Model Export](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- [Circuit Generation with GANs Research Paper](https://arxiv.org/abs/example)
- [Diffusion Models for Circuit Design](https://arxiv.org/abs/example)

## Review History

- **2025-01-31**: Initial proposal and team review
- **2025-01-31**: Final approval and acceptance

---

*This ADR was created on 2025-01-31 and approved by the technical team.*