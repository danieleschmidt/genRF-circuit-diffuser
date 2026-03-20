# genRF-circuit-diffuser

Generative AI for analog/RF circuit design via diffusion-inspired parameter sampling and SPICE simulation.

## Components

- **CircuitTopology** – Nodes and components: R, L, C, transistor
- **SPICESimulator** – AC frequency analysis using impedance methods
- **DiffusionDesigner** – Gaussian noise + Langevin denoising to sample circuit parameters
- **PerformanceEvaluator** – Bandwidth, gain, impedance match scoring

## Usage

```python
from genrf.topology import CircuitTopology
from genrf.spice_sim import SPICESimulator
from genrf.diffusion_designer import DiffusionDesigner
from genrf.performance_evaluator import PerformanceEvaluator

# Build an LC filter
topo = CircuitTopology.lc_filter(L=1e-9, C=1e-12, R_load=50.0)

# Simulate
sim = SPICESimulator()
result = sim.simulate(topo)
print(f"Peak gain: {result['magnitude_db'].max():.1f} dB")

# Evaluate performance
evaluator = PerformanceEvaluator(target_freq=1e9)
metrics = evaluator.evaluate(topo)
print(f"Score: {metrics['score']:.3f}, BW: {metrics['bandwidth']/1e6:.1f} MHz")

# Sample new circuits via diffusion
designer = DiffusionDesigner(n_steps=50)
samples = designer.sample_random(topo, n_samples=10)
```

## Development

```bash
pip install numpy
pytest tests/
```
