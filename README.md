# genRF-circuit-diffuser

Generative AI for analog/RF circuit design using diffusion models.

## Features

- **CircuitTopology**: R/L/C component graph with SPICE-like netlist generation
- **SPICESimulator**: AC analysis via impedance calculations (1kHz–1GHz)
- **DiffusionDesigner**: noise-to-circuit parameter generation
- **PerformanceEvaluator**: bandwidth, gain, Q-factor metrics
- **RFCircuitGenerator**: spec-driven circuit synthesis

## Usage

```python
from genrf.generator import RFCircuitGenerator

gen = RFCircuitGenerator()
circuit = gen.best_match({"bandwidth_hz": 1e6, "gain_db": 0, "topology": "lowpass"})
print(circuit.get_netlist())
```

## Design Specifications

- Topology: lowpass, highpass, bandpass
- Frequency range: 1kHz to 1GHz
- Component types: R, L, C
