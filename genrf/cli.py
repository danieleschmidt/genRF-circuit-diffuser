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
    print(f"Generating circuit from {args.spec}")
    print(f"Output: {args.output} (format: {args.format})")
    # TODO: Implement circuit generation
    return 0


def _launch_dashboard(args):
    """Launch interactive dashboard."""
    print(f"Starting dashboard at http://{args.host}:{args.port}")
    # TODO: Implement dashboard launch
    return 0


if __name__ == "__main__":
    sys.exit(main())