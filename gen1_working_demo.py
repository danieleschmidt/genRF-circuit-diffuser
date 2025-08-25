#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK (Simple) - Basic functional demonstration

This demo tests the core GenRF circuit generation functionality with minimal working features.
"""

import sys
import os
import time
import warnings
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def activate_venv():
    """Activate the virtual environment if it exists."""
    venv_path = Path(__file__).parent / "genrf_env"
    if venv_path.exists():
        activate_script = venv_path / "bin" / "activate_this.py"
        if activate_script.exists():
            with open(activate_script) as f:
                exec(f.read(), {'__file__': str(activate_script)})
        else:
            # Manually adjust sys.path for the venv
            site_packages = venv_path / "lib" / "python3.12" / "site-packages"
            if site_packages.exists():
                sys.path.insert(0, str(site_packages))

# Activate virtual environment
activate_venv()

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    """Main execution function for Generation 1 demo."""
    print("🚀 GenRF Generation 1: MAKE IT WORK (Simple)")
    print("=" * 50)
    
    try:
        # Step 1: Test basic package import
        print("📦 Testing package import...")
        import genrf
        print(f"✅ GenRF v{genrf.__version__} imported successfully")
        
        # Step 2: Test basic classes
        print("\n🔧 Testing core components...")
        from genrf import DesignSpec, CircuitDiffuser
        print("✅ Core classes imported successfully")
        
        # Step 3: Create a simple design specification
        print("\n📋 Creating design specification...")
        spec = DesignSpec(
            circuit_type="LNA",
            frequency=2.4e9,  # 2.4 GHz
            name="Demo_LNA",
            description="Simple LNA for Generation 1 demo"
        )
        print(f"✅ Design spec created: {spec.circuit_type} at {spec.frequency/1e9:.1f} GHz")
        
        # Step 4: Initialize the circuit diffuser (minimal config)
        print("\n🧠 Initializing circuit diffuser...")
        diffuser = CircuitDiffuser(
            device="cpu",  # Use CPU for compatibility
            verbose=True
        )
        print("✅ CircuitDiffuser initialized successfully")
        
        # Step 5: Test basic parameter generation (mock implementation)
        print("\n⚡ Testing basic circuit generation...")
        start_time = time.time()
        
        # Generate a simple circuit (this will use mock/demo data initially)
        result = diffuser.generate_simple(spec, n_candidates=1)
        
        generation_time = time.time() - start_time
        print(f"✅ Circuit generated in {generation_time:.2f}s")
        
        # Step 6: Display results
        print("\n📊 Generation Results:")
        print(f"  • Netlist length: {len(result.netlist)} characters")
        print(f"  • Parameters: {len(result.parameters)} components")
        print(f"  • Topology: {result.topology}")
        print(f"  • Technology: {result.technology}")
        print(f"  • SPICE valid: {result.spice_valid}")
        
        if result.performance:
            print(f"  • Performance metrics: {len(result.performance)} values")
            if 'gain_db' in result.performance:
                print(f"    - Gain: {result.gain:.1f} dB")
        
        # Step 7: Test export functionality
        print("\n💾 Testing export capabilities...")
        output_dir = Path("gen1_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Export netlist
        netlist_file = output_dir / "demo_lna_netlist.spice"
        with open(netlist_file, 'w') as f:
            f.write(result.netlist)
        print(f"✅ Netlist exported to: {netlist_file}")
        
        # Export parameters
        import json
        params_file = output_dir / "demo_lna_params.json"
        export_data = {
            "design_spec": {
                "circuit_type": spec.circuit_type,
                "frequency": spec.frequency,
                "name": spec.name
            },
            "parameters": result.parameters,
            "performance": result.performance,
            "topology": result.topology,
            "generation_time": generation_time
        }
        
        with open(params_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✅ Parameters exported to: {params_file}")
        
        print("\n🎉 Generation 1 Demo Completed Successfully!")
        print(f"📁 Outputs saved to: {output_dir}")
        print(f"⏱️  Total execution time: {time.time() - start_time:.2f}s")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: source genrf_env/bin/activate")
        return False
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)