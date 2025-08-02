# GenRF Circuit Diffuser Architecture

## System Overview

GenRF Circuit Diffuser is a sophisticated AI-driven system for automatic analog and RF circuit design generation using generative models and SPICE simulation.

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        CLI[CLI Interface]
        API[REST API]
        Dashboard[Grafana Dashboard]
    end
    
    subgraph "Application Layer"
        CircuitGen[Circuit Generator]
        SpecParser[Specification Parser]
        Optimizer[Multi-Objective Optimizer]
        Exporter[Code Exporter]
    end
    
    subgraph "AI/ML Layer"
        CycleGAN[Topology CycleGAN]
        Diffuser[Parameter Diffuser]
        BayesOpt[Bayesian Optimizer]
    end
    
    subgraph "Simulation Layer"
        SPICE[SPICE Engine]
        Validator[Circuit Validator]
        MonteCarloSim[Monte Carlo Analysis]
    end
    
    subgraph "Data Layer"
        ModelStore[Model Storage]
        CircuitDB[Circuit Database]
        TechFiles[Technology Files]
    end
    
    CLI --> CircuitGen
    API --> CircuitGen
    Dashboard --> CircuitGen
    
    CircuitGen --> SpecParser
    CircuitGen --> CycleGAN
    CircuitGen --> Diffuser
    
    CycleGAN --> SPICE
    Diffuser --> SPICE
    SPICE --> Validator
    Validator --> BayesOpt
    BayesOpt --> Optimizer
    
    Optimizer --> Exporter
    Exporter --> TechFiles
    
    MonteCarloSim --> SPICE
    
    ModelStore --> CycleGAN
    ModelStore --> Diffuser
    CircuitDB --> CycleGAN
    TechFiles --> SPICE
```

## Core Components

### 1. Circuit Generator (`genrf.core.CircuitDiffuser`)
- **Purpose**: Main orchestrator for the circuit generation pipeline
- **Responsibilities**:
  - Coordinate between topology generation and parameter optimization
  - Manage the iterative SPICE-in-the-loop process
  - Handle multi-objective optimization constraints
- **Key Interfaces**: `generate()`, `optimize_for_yield()`, `generate_chain()`

### 2. Topology Generator (`genrf.models.CycleGAN`)
- **Purpose**: Generate circuit topologies using cycle-consistent GANs
- **Training Data**: 50k+ production circuit topologies
- **Supported Types**: LNA, Mixer, VCO, PA, Filter circuits
- **Output**: Netlist structure with component connectivity

### 3. Parameter Diffuser (`genrf.models.DiffusionModel`)
- **Purpose**: Optimize component values using denoising diffusion
- **Architecture**: U-Net based diffusion model
- **Conditioning**: Technology constraints, specification targets
- **Output**: Component values (resistors, capacitors, transistor sizes)

### 4. SPICE Integration (`genrf.simulation`)
- **Supported Engines**: NgSpice, XYCE, Spectre
- **Analysis Types**: AC, DC, Transient, Noise, Monte Carlo
- **Technology Support**: Multiple PDKs (TSMC, GlobalFoundries, etc.)

### 5. Multi-Objective Optimizer (`genrf.optimization`)
- **Algorithm**: Bayesian optimization with Pareto front exploration
- **Objectives**: Gain, noise figure, power consumption, area
- **Constraints**: Technology rules, specification requirements

## Data Flow

1. **Input Processing**
   - User specification → DesignSpec object
   - Technology file loading → TechnologyFile object
   - Constraint validation

2. **Topology Generation**
   - CycleGAN generates initial topology candidates
   - Topology validation against design rules
   - Connectivity analysis

3. **Parameter Optimization**
   - Diffusion model generates component values
   - SPICE simulation for performance evaluation
   - Iterative refinement based on objectives

4. **Validation & Export**
   - Final SPICE verification
   - Monte Carlo yield analysis
   - Code generation (SKILL, Verilog-A, ADS)

## Technology Integration

### PDK Support
- **TSMC**: 65nm, 40nm, 28nm, 16nm, 7nm
- **GlobalFoundries**: 22FDX, 14nm, 12nm
- **Custom**: Configurable technology files

### EDA Tool Integration
- **Cadence**: SKILL script generation, Ocean testbench
- **Keysight ADS**: Netlist export, layout hints
- **Synopsys**: Verilog-A model generation
- **KLayout**: Layout constraint generation

## Performance Characteristics

### Generation Speed
- Topology generation: ~10 seconds
- Parameter optimization: 1-5 minutes
- Full multi-candidate optimization: 5-30 minutes

### Quality Metrics
- SPICE accuracy: 95%+ correlation with silicon
- Design time reduction: 500-800x vs manual design
- Performance improvement: 7% average FoM increase

## Scalability Considerations

### Horizontal Scaling
- Model inference can be distributed across GPUs
- SPICE simulations parallelizable across cores
- Bayesian optimization supports batch evaluation

### Memory Requirements
- Model storage: ~2GB for full model set
- Runtime memory: 4-8GB depending on circuit complexity
- SPICE simulation memory scales with circuit size

## Security & Compliance

### Model Security
- Models stored with integrity checksums
- Version control for model updates
- Audit trail for generated designs

### IP Protection
- Technology file encryption support
- Generated design watermarking
- Access control for proprietary PDKs

## Monitoring & Observability

### Key Metrics
- Generation success rate
- SPICE simulation convergence rate
- Performance objective achievement
- Resource utilization (CPU, GPU, memory)

### Logging
- Structured logging with correlation IDs
- Performance metrics collection
- Error tracking and alerting
- Design generation audit trail

## Extension Points

### Custom Circuit Types
- Pluggable circuit type definitions
- Custom objective functions
- Domain-specific constraints

### Technology Integration
- Custom PDK adapters
- Simulation engine plugins
- Layout tool integrations

### Model Training
- Transfer learning for new technologies
- Custom loss functions
- Domain adaptation techniques