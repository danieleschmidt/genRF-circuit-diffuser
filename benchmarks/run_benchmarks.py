#!/usr/bin/env python3
"""
Benchmark runner for GenRF Circuit Diffuser.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --config benchmarks/benchmark_config.yaml
    python benchmarks/run_benchmarks.py --scenario LNA_2G4
"""

import argparse
import json
import time
import psutil
import yaml
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Performance benchmark runner."""
    
    def __init__(self, config_path: str = "benchmarks/benchmark_config.yaml"):
        """Initialize benchmark runner."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load benchmark configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default benchmark configuration."""
        return {
            'performance_targets': {
                'circuit_generation': {'time_limit_seconds': 5.0},
                'spice_simulation': {'time_limit_seconds': 2.0}
            },
            'benchmark_suite': {
                'warmup_iterations': 3,
                'measurement_iterations': 10
            }
        }
    
    def run_all_benchmarks(self):
        """Run all configured benchmarks."""
        logger.info("Starting benchmark suite...")
        
        # Get system information
        system_info = self._get_system_info()
        
        # Run benchmarks for each scenario
        scenarios = self.config.get('test_scenarios', [])
        if not scenarios:
            scenarios = [{'name': 'default', 'circuit_type': 'LNA'}]
        
        for scenario in scenarios:
            logger.info(f"Running benchmark for scenario: {scenario['name']}")
            result = self._run_scenario_benchmark(scenario)
            self.results.append(result)
        
        # Save results
        self._save_results(system_info)
        
        # Check for regressions
        self._check_regressions()
        
        logger.info("Benchmark suite completed")
    
    def _run_scenario_benchmark(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run benchmark for a specific scenario."""
        scenario_name = scenario['name']
        warmup_iterations = self.config['benchmark_suite']['warmup_iterations']
        measurement_iterations = self.config['benchmark_suite']['measurement_iterations']
        
        # Warmup runs
        logger.info(f"Running {warmup_iterations} warmup iterations...")
        for _ in range(warmup_iterations):
            self._simulate_circuit_generation(scenario)
        
        # Measurement runs
        logger.info(f"Running {measurement_iterations} measurement iterations...")
        times = []
        memory_usage = []
        
        for i in range(measurement_iterations):
            start_memory = psutil.Process().memory_info().rss
            start_time = time.perf_counter()
            
            self._simulate_circuit_generation(scenario)
            
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            times.append(execution_time)
            memory_usage.append(memory_used)
            
            logger.info(f"  Iteration {i+1}: {execution_time:.3f}s, "
                       f"{memory_used/1024/1024:.1f}MB")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_memory = sum(memory_usage) / len(memory_usage)
        
        return {
            'scenario': scenario_name,
            'circuit_type': scenario.get('circuit_type', 'unknown'),
            'measurements': {
                'execution_time': {
                    'average': avg_time,
                    'minimum': min_time,
                    'maximum': max_time,
                    'samples': times
                },
                'memory_usage': {
                    'average': avg_memory,
                    'samples': memory_usage
                }
            },
            'timestamp': time.time()
        }
    
    def _simulate_circuit_generation(self, scenario: Dict[str, Any]):
        """Simulate circuit generation (mock for now)."""
        # This would call actual GenRF circuit generation
        circuit_type = scenario.get('circuit_type', 'LNA')
        complexity = scenario.get('complexity', 'medium')
        
        # Simulate different processing times based on complexity
        if complexity == 'low':
            time.sleep(0.1)
        elif complexity == 'medium':
            time.sleep(0.2)
        else:  # high
            time.sleep(0.4)
        
        # Simulate memory allocation
        dummy_data = [0] * (1000000 if complexity == 'high' else 500000)
        del dummy_data
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'python_version': __import__('sys').version,
            'platform': __import__('platform').platform(),
            'timestamp': time.time()
        }
    
    def _save_results(self, system_info: Dict[str, Any]):
        """Save benchmark results to file."""
        output_config = self.config.get('output', {})
        output_path = Path(output_config.get('file_path', 'benchmark_results.json'))
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = {
            'system_info': system_info,
            'config': self.config,
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary."""
        if not self.results:
            return {}
        
        total_scenarios = len(self.results)
        avg_generation_time = sum(
            r['measurements']['execution_time']['average'] 
            for r in self.results
        ) / total_scenarios
        
        return {
            'total_scenarios': total_scenarios,
            'average_generation_time': avg_generation_time,
            'fastest_scenario': min(
                self.results, 
                key=lambda x: x['measurements']['execution_time']['average']
            )['scenario'],
            'slowest_scenario': max(
                self.results,
                key=lambda x: x['measurements']['execution_time']['average']
            )['scenario']
        }
    
    def _check_regressions(self):
        """Check for performance regressions."""
        # This would compare with baseline results
        logger.info("Checking for performance regressions...")
        # Implementation would load baseline and compare
        pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run GenRF performance benchmarks")
    parser.add_argument(
        '--config', 
        default='benchmarks/benchmark_config.yaml',
        help='Path to benchmark configuration file'
    )
    parser.add_argument(
        '--scenario',
        help='Run specific scenario only'
    )
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.config)
    
    if args.scenario:
        # Run specific scenario
        scenarios = runner.config.get('test_scenarios', [])
        scenario = next((s for s in scenarios if s['name'] == args.scenario), None)
        if scenario:
            result = runner._run_scenario_benchmark(scenario)
            print(json.dumps(result, indent=2))
        else:
            logger.error(f"Scenario '{args.scenario}' not found")
            return 1
    else:
        # Run all benchmarks
        runner.run_all_benchmarks()
    
    return 0


if __name__ == '__main__':
    exit(main())