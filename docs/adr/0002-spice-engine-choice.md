# ADR-0002: SPICE Engine Selection

## Status
Accepted

## Context
The circuit generator requires a SPICE simulation engine for:
- Validating generated circuit performance
- Providing feedback to optimization algorithms
- Ensuring physical accuracy of designs
- Supporting multiple technology nodes

Key requirements:
- High simulation accuracy for RF circuits
- Python integration capabilities
- Support for multiple PDKs
- Performance suitable for optimization loops

## Decision
Primary engine: **NgSpice** with **PySpice** wrapper
Secondary support: **XYCE** for advanced RF analysis

### NgSpice Selection Rationale
1. **Open Source**: No licensing restrictions for research and commercial use
2. **RF Capabilities**: Strong AC, noise, and RF analysis support
3. **Python Integration**: Mature PySpice wrapper with good API
4. **PDK Support**: Compatible with most foundry models
5. **Performance**: Adequate speed for optimization loops
6. **Community**: Active development and extensive documentation

### XYCE Secondary Support
1. **Advanced RF**: Superior RF and microwave capabilities
2. **Parallel Simulation**: Built-in parallelization support
3. **Sandia Heritage**: Strong numerical algorithms
4. **Research Focus**: Cutting-edge simulation techniques

## Consequences

### Positive
- NgSpice provides reliable baseline simulation
- PySpice offers excellent Python integration
- XYCE available for advanced RF scenarios
- Open source eliminates licensing concerns
- Good community support and documentation

### Negative
- NgSpice performance may limit optimization speed
- XYCE integration more complex than NgSpice
- May require custom model translation for some PDKs
- Multiple engines increase testing complexity

## Implementation Details

### PySpice Integration
```python
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Spice.Netlist import Circuit

class SPICESimulator:
    def __init__(self, engine="ngspice"):
        if engine == "ngspice":
            self.simulator = NgSpiceShared.new_instance()
        elif engine == "xyce":
            self.simulator = XyceInterface()
```

### Performance Considerations
- Batch simulations for parallel evaluation
- Circuit complexity limits for real-time optimization
- Result caching for repeated parameter sweeps

## Alternatives Considered

### Spectre (Cadence)
- **Pros**: Industry standard, excellent RF accuracy
- **Cons**: Expensive licensing, limited Python integration

### HSPICE (Synopsys)
- **Pros**: High accuracy, industry adoption
- **Cons**: Licensing costs, integration complexity

### LTspice
- **Pros**: Free, good performance
- **Cons**: Limited Python integration, Windows-centric

### QucsStudio
- **Pros**: Modern interface, good RF support
- **Cons**: Newer tool, smaller community

## Related ADRs
- [ADR-0001](./0001-ai-model-architecture.md) - AI Model Architecture Selection
- [ADR-0003](./0003-technology-file-format.md) - Technology File Format Standard