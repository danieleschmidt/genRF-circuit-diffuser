# ADR-0003: Technology File Format Standard

## Status
Accepted

## Context
The circuit generator needs a standardized way to represent technology information including:
- Device models and parameters
- Design rules and constraints
- Process variation data
- Layout information
- Foundry-specific parameters

Requirements:
- Support multiple foundries (TSMC, GlobalFoundries, etc.)
- Version control and compatibility tracking
- Extensible for new technologies
- Human-readable for debugging
- Machine-parseable for automation

## Decision
Use **YAML-based technology files** with the following structure:

```yaml
technology:
  name: "TSMC65nm_RF"
  version: "1.2.0"
  foundry: "TSMC"
  node: "65nm"
  
models:
  transistors:
    nfet:
      model_file: "models/nch_rf.lib"
      parameters:
        vth0: 0.35
        toxe: 2.3e-9
    pfet:
      model_file: "models/pch_rf.lib"
      parameters:
        vth0: -0.35
        toxe: 2.3e-9
        
  passives:
    resistors:
      - type: "poly"
        sheet_resistance: 350
        model_file: "models/rpoly.lib"
    capacitors:
      - type: "mim"
        capacitance_per_area: 1.5e-15
        model_file: "models/cmim.lib"

design_rules:
  minimum_channel_length: 60e-9
  minimum_channel_width: 120e-9
  maximum_vdd: 1.2
  substrate:
    type: "bulk"
    resistivity: 10
    
process_corners:
  - name: "TT"
    temperature: 27
    voltage: 1.2
  - name: "FF"
    temperature: -40
    voltage: 1.32
  - name: "SS"
    temperature: 125
    voltage: 1.08

layout:
  metal_layers: 9
  via_rules:
    minimum_via_size: 100e-9
  density_rules:
    minimum_density: 0.3
    maximum_density: 0.85
```

## Consequences

### Positive
- YAML is human-readable and widely supported
- Hierarchical structure naturally represents technology data
- Version control friendly (text-based)
- Easy to extend for new parameters
- Supports comments for documentation
- Standard YAML libraries available in all languages

### Negative
- YAML parsing overhead compared to binary formats
- Requires validation schema to ensure correctness
- Large files for complex technologies
- May need compression for distribution

## Implementation Details

### Validation Schema
Use JSON Schema to validate technology files:

```python
import yaml
import jsonschema

class TechnologyFile:
    def __init__(self, filepath):
        with open(filepath) as f:
            self.data = yaml.safe_load(f)
        self._validate()
    
    def _validate(self):
        with open('schemas/technology.schema.json') as f:
            schema = json.load(f)
        jsonschema.validate(self.data, schema)
```

### Caching Strategy
- Parse and cache technology files at startup
- Detect file changes for automatic reloading
- Memory-efficient storage of parsed data

## Alternatives Considered

### JSON Format
- **Pros**: Wide language support, standard web format
- **Cons**: No comments, less human-readable

### XML Format
- **Pros**: Schema validation, namespace support
- **Cons**: Verbose, complex parsing

### Binary Format (Protocol Buffers)
- **Pros**: Compact, fast parsing, strong typing
- **Cons**: Not human-readable, requires special tools

### Python Configuration Files
- **Pros**: Full programming language features
- **Cons**: Security risks, platform-specific

## Related ADRs
- [ADR-0002](./0002-spice-engine-choice.md) - SPICE Engine Selection