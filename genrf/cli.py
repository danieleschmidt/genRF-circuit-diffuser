"""Command-line interface for GenRF circuit diffuser."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GenRF Circuit Diffuser - AI-powered RF circuit generation"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"genrf {__import__('genrf').__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate RF circuit")
    gen_parser.add_argument("--spec", required=True, help="Design specification file")
    gen_parser.add_argument("--output", default="output", help="Output directory")
    gen_parser.add_argument("--format", choices=["skill", "verilog-a", "ads"], 
                           default="skill", help="Output format")
    
    # Dashboard command  
    dash_parser = subparsers.add_parser("dashboard", help="Launch design dashboard")
    dash_parser.add_argument("--port", type=int, default=3000, help="Dashboard port")
    dash_parser.add_argument("--host", default="localhost", help="Dashboard host")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    try:
        if args.command == "generate":
            return _generate_circuit(args)
        elif args.command == "dashboard":
            return _launch_dashboard(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
        
    return 0


def _generate_circuit(args):
    """Generate circuit from specification."""
    try:
        from . import CircuitDiffuser, DesignSpec, TechnologyFile, CodeExporter
        import yaml
        import json
        
        print(f"Loading specification from {args.spec}")
        
        # Load design specification
        spec_path = Path(args.spec)
        if not spec_path.exists():
            print(f"Error: Specification file not found: {args.spec}", file=sys.stderr)
            return 1
        
        # Load spec based on file extension
        if spec_path.suffix.lower() == '.yaml' or spec_path.suffix.lower() == '.yml':
            spec = DesignSpec.from_yaml(spec_path)
        elif spec_path.suffix.lower() == '.json':
            spec = DesignSpec.from_json(spec_path)
        else:
            print(f"Error: Unsupported specification format: {spec_path.suffix}", file=sys.stderr)
            return 1
        
        print(f"Loaded specification: {spec.name}")
        print(f"Circuit type: {spec.circuit_type}")
        print(f"Frequency: {spec.frequency/1e9:.2f} GHz")
        
        # Initialize circuit diffuser
        print("Initializing CircuitDiffuser...")
        diffuser = CircuitDiffuser(
            spice_engine="ngspice",
            technology=TechnologyFile.default(),
            verbose=True
        )
        
        # Generate circuit
        print("Generating circuit...")
        circuit = diffuser.generate(
            spec,
            n_candidates=5,  # Reduced for demo
            optimization_steps=10,  # Reduced for demo
            validate_spice=False  # Skip SPICE validation for now
        )
        
        print(f"Generation complete!")
        print(f"Best performance: Gain={circuit.gain:.1f}dB, NF={circuit.nf:.2f}dB, Power={circuit.power*1000:.1f}mW")
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Export circuit
        exporter = CodeExporter()
        
        if args.format == 'skill':
            output_file = output_dir / f"{spec.name}.il"
            circuit.export_skill(output_file)
        elif args.format == 'verilog-a':
            output_file = output_dir / f"{spec.name}.va"
            circuit.export_verilog_a(output_file)
        elif args.format == 'ads':
            output_file = output_dir / f"{spec.name}.net"
            exporter.export_ads(circuit, output_file)
        else:
            # Export all formats
            exported_files = exporter.export_all_formats(circuit, output_dir)
            print(f"Exported {len(exported_files)} formats:")
            for fmt, filepath in exported_files.items():
                print(f"  {fmt}: {filepath}")
            return 0
        
        print(f"Circuit exported to: {output_file}")
        return 0
        
    except ImportError as e:
        print(f"Import Error: {e}", file=sys.stderr)
        print("Please ensure all dependencies are installed.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error generating circuit: {e}", file=sys.stderr)
        return 1


def _launch_dashboard(args):
    """Launch interactive dashboard."""
    try:
        import threading
        import time
        
        print(f"Starting GenRF dashboard at http://{args.host}:{args.port}")
        print("Dashboard functionality coming soon!")
        print("For now, use the 'generate' command to create circuits.")
        
        # Placeholder implementation
        print("Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
            return 0
            
    except Exception as e:
        print(f"Error launching dashboard: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())