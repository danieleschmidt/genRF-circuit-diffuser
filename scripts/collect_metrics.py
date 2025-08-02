#!/usr/bin/env python3
"""
Automated metrics collection script for GenRF Circuit Diffuser.

This script collects various metrics from different sources and updates
the project metrics tracking system.
"""
import json
import os
import subprocess
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
from dataclasses import dataclass, asdict


@dataclass
class MetricValue:
    """Represents a metric value with metadata."""
    value: Any
    timestamp: str
    source: str
    unit: Optional[str] = None
    trend: Optional[str] = None


class MetricsCollector:
    """Main metrics collection and management class."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.metrics = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load metrics configuration."""
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {self.config_path}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config: {e}")
            return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all configured metrics."""
        self.logger.info("Starting metrics collection")
        
        # Development metrics
        self.collect_code_quality_metrics()
        self.collect_performance_metrics()
        self.collect_reliability_metrics()
        
        # Business metrics
        self.collect_adoption_metrics()
        self.collect_usage_metrics()
        self.collect_quality_metrics()
        
        # Infrastructure metrics
        self.collect_cicd_metrics()
        self.collect_monitoring_metrics()
        self.collect_security_metrics()
        
        self.logger.info("Metrics collection completed")
        return self.metrics
    
    def collect_code_quality_metrics(self):
        """Collect code quality metrics."""
        self.logger.info("Collecting code quality metrics")
        
        # Test coverage
        coverage = self._get_test_coverage()
        if coverage is not None:
            self.metrics['test_coverage'] = MetricValue(
                value=coverage,
                timestamp=datetime.now().isoformat(),
                source="pytest-cov",
                unit="percentage"
            )
        
        # Code complexity
        complexity = self._get_code_complexity()
        if complexity is not None:
            self.metrics['code_complexity'] = MetricValue(
                value=complexity,
                timestamp=datetime.now().isoformat(),
                source="radon",
                unit="cyclomatic_complexity"
            )
        
        # Documentation coverage
        doc_coverage = self._get_documentation_coverage()
        if doc_coverage is not None:
            self.metrics['documentation_coverage'] = MetricValue(
                value=doc_coverage,
                timestamp=datetime.now().isoformat(),
                source="sphinx",
                unit="percentage"
            )
        
        # Security vulnerabilities
        vulnerabilities = self._get_security_vulnerabilities()
        if vulnerabilities is not None:
            self.metrics['security_vulnerabilities'] = MetricValue(
                value=vulnerabilities,
                timestamp=datetime.now().isoformat(),
                source="safety",
                unit="count"
            )
    
    def collect_performance_metrics(self):
        """Collect performance metrics."""
        self.logger.info("Collecting performance metrics")
        
        # Run benchmark tests if available
        benchmark_results = self._run_performance_benchmarks()
        
        if benchmark_results:
            for metric_name, value in benchmark_results.items():
                self.metrics[metric_name] = MetricValue(
                    value=value,
                    timestamp=datetime.now().isoformat(),
                    source="pytest-benchmark",
                    unit="seconds"
                )
    
    def collect_reliability_metrics(self):
        """Collect reliability metrics."""
        self.logger.info("Collecting reliability metrics")
        
        # Check if we have historical data for uptime calculation
        uptime = self._calculate_uptime()
        if uptime is not None:
            self.metrics['uptime'] = MetricValue(
                value=uptime,
                timestamp=datetime.now().isoformat(),
                source="monitoring",
                unit="percentage"
            )
    
    def collect_adoption_metrics(self):
        """Collect adoption and community metrics."""
        self.logger.info("Collecting adoption metrics")
        
        # GitHub stars
        github_token = os.getenv('GITHUB_TOKEN')
        if github_token:
            stars = self._get_github_stars(github_token)
            if stars is not None:
                self.metrics['github_stars'] = MetricValue(
                    value=stars,
                    timestamp=datetime.now().isoformat(),
                    source="github_api",
                    unit="count"
                )
        
        # PyPI downloads
        downloads = self._get_pypi_downloads()
        if downloads is not None:
            self.metrics['downloads'] = MetricValue(
                value=downloads,
                timestamp=datetime.now().isoformat(),
                source="pypistats",
                unit="monthly_downloads"
            )
    
    def collect_usage_metrics(self):
        """Collect usage metrics."""
        self.logger.info("Collecting usage metrics")
        
        # This would typically connect to application analytics
        # For now, we'll use placeholder values
        pass
    
    def collect_quality_metrics(self):
        """Collect quality metrics."""
        self.logger.info("Collecting quality metrics")
        
        # Circuit quality scores would come from application logs
        # For now, we'll use placeholder values
        pass
    
    def collect_cicd_metrics(self):
        """Collect CI/CD metrics."""
        self.logger.info("Collecting CI/CD metrics")
        
        github_token = os.getenv('GITHUB_TOKEN')
        if github_token:
            build_success_rate = self._get_build_success_rate(github_token)
            if build_success_rate is not None:
                self.metrics['build_success_rate'] = MetricValue(
                    value=build_success_rate,
                    timestamp=datetime.now().isoformat(),
                    source="github_actions",
                    unit="percentage"
                )
    
    def collect_monitoring_metrics(self):
        """Collect monitoring metrics."""
        self.logger.info("Collecting monitoring metrics")
        
        # This would query Prometheus or other monitoring systems
        pass
    
    def collect_security_metrics(self):
        """Collect security metrics."""
        self.logger.info("Collecting security metrics")
        
        # Check for dependency vulnerabilities
        dep_vulns = self._check_dependency_vulnerabilities()
        if dep_vulns is not None:
            self.metrics['dependency_vulnerabilities'] = MetricValue(
                value=dep_vulns,
                timestamp=datetime.now().isoformat(),
                source="safety",
                unit="count"
            )
    
    def _get_test_coverage(self) -> Optional[float]:
        """Get test coverage percentage."""
        try:
            # Run pytest with coverage
            result = subprocess.run([
                'python', '-m', 'pytest', '--cov=genrf', '--cov-report=json'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse coverage.json if it exists
                coverage_file = Path('coverage.json')
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                        return coverage_data.get('totals', {}).get('percent_covered')
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            self.logger.warning("Could not collect test coverage")
        return None
    
    def _get_code_complexity(self) -> Optional[float]:
        """Get average code complexity."""
        try:
            result = subprocess.run([
                'radon', 'cc', 'genrf/', '-j'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                complexities = []
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if 'complexity' in item:
                            complexities.append(item['complexity'])
                
                if complexities:
                    return sum(complexities) / len(complexities)
        except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
            self.logger.warning("Could not collect code complexity")
        return None
    
    def _get_documentation_coverage(self) -> Optional[float]:
        """Get documentation coverage percentage."""
        try:
            # This is a simplified check - in practice, you'd use tools like
            # interrogate or pydocstyle
            python_files = list(Path('genrf').rglob('*.py'))
            documented_files = 0
            
            for file_path in python_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Simple check for docstrings
                    if '"""' in content or "'''" in content:
                        documented_files += 1
            
            if python_files:
                return (documented_files / len(python_files)) * 100
        except Exception:
            self.logger.warning("Could not collect documentation coverage")
        return None
    
    def _get_security_vulnerabilities(self) -> Optional[int]:
        """Get count of security vulnerabilities."""
        try:
            result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                safety_data = json.loads(result.stdout)
                return len(safety_data)
            else:
                # safety returns non-zero when vulnerabilities are found
                try:
                    safety_data = json.loads(result.stdout)
                    return len(safety_data)
                except json.JSONDecodeError:
                    return 0
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.warning("Could not collect security vulnerabilities")
        return None
    
    def _run_performance_benchmarks(self) -> Optional[Dict[str, float]]:
        """Run performance benchmarks and return results."""
        try:
            result = subprocess.run([
                'python', '-m', 'pytest', 'benchmarks/', '--benchmark-json=benchmark.json'
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                benchmark_file = Path('benchmark.json')
                if benchmark_file.exists():
                    with open(benchmark_file) as f:
                        benchmark_data = json.load(f)
                        
                    results = {}
                    for benchmark in benchmark_data.get('benchmarks', []):
                        name = benchmark['name']
                        stats = benchmark['stats']
                        results[f"{name}_mean"] = stats['mean']
                        results[f"{name}_stddev"] = stats['stddev']
                    
                    return results
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            self.logger.warning("Could not run performance benchmarks")
        return None
    
    def _calculate_uptime(self) -> Optional[float]:
        """Calculate system uptime percentage."""
        # This would typically query monitoring systems like Prometheus
        # For now, return a placeholder
        return None
    
    def _get_github_stars(self, token: str) -> Optional[int]:
        """Get GitHub repository star count."""
        try:
            repo = self.config.get('project', {}).get('repository')
            if not repo:
                return None
            
            url = f"https://api.github.com/repos/{repo}"
            headers = {
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get('stargazers_count')
        except requests.RequestException:
            self.logger.warning("Could not fetch GitHub stars")
        return None
    
    def _get_pypi_downloads(self) -> Optional[int]:
        """Get PyPI download count."""
        try:
            # This would use pypistats API
            # For now, return None
            return None
        except Exception:
            self.logger.warning("Could not fetch PyPI downloads")
        return None
    
    def _get_build_success_rate(self, token: str) -> Optional[float]:
        """Get CI/CD build success rate."""
        try:
            repo = self.config.get('project', {}).get('repository')
            if not repo:
                return None
            
            # Get recent workflow runs
            url = f"https://api.github.com/repos/{repo}/actions/runs"
            headers = {
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            params = {
                'per_page': 100,
                'created': f">={(datetime.now() - timedelta(days=30)).isoformat()}"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                runs = data.get('workflow_runs', [])
                
                if runs:
                    successful_runs = sum(1 for run in runs if run['conclusion'] == 'success')
                    return (successful_runs / len(runs)) * 100
        except requests.RequestException:
            self.logger.warning("Could not fetch build success rate")
        return None
    
    def _check_dependency_vulnerabilities(self) -> Optional[int]:
        """Check for dependency vulnerabilities."""
        return self._get_security_vulnerabilities()
    
    def update_config(self):
        """Update the metrics configuration with collected values."""
        timestamp = datetime.now().isoformat()
        
        for metric_name, metric_value in self.metrics.items():
            # Navigate the nested config structure to update values
            self._update_nested_metric(self.config, metric_name, metric_value.value, timestamp)
        
        # Save updated configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Updated {len(self.metrics)} metrics in configuration")
    
    def _update_nested_metric(self, config: Dict, metric_name: str, value: Any, timestamp: str):
        """Update a metric value in the nested configuration structure."""
        # This is a simplified update - in practice, you'd need more sophisticated
        # logic to find and update metrics in the nested structure
        
        def find_and_update(obj, name, val, ts):
            if isinstance(obj, dict):
                for key, item in obj.items():
                    if key == name and isinstance(item, dict) and 'current' in item:
                        item['current'] = val
                        item['last_updated'] = ts
                        return True
                    elif isinstance(item, dict):
                        if find_and_update(item, name, val, ts):
                            return True
            return False
        
        find_and_update(config, metric_name, value, timestamp)
    
    def generate_report(self, output_format: str = 'json') -> str:
        """Generate a metrics report."""
        if output_format == 'json':
            return json.dumps({
                'timestamp': datetime.now().isoformat(),
                'metrics': {k: asdict(v) for k, v in self.metrics.items()}
            }, indent=2)
        elif output_format == 'markdown':
            return self._generate_markdown_report()
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_markdown_report(self) -> str:
        """Generate a markdown metrics report."""
        lines = [
            "# GenRF Circuit Diffuser Metrics Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            ""
        ]
        
        for metric_name, metric_value in self.metrics.items():
            lines.extend([
                f"### {metric_name.replace('_', ' ').title()}",
                f"- **Value**: {metric_value.value} {metric_value.unit or ''}",
                f"- **Source**: {metric_value.source}",
                f"- **Timestamp**: {metric_value.timestamp}",
                ""
            ])
        
        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect GenRF project metrics")
    parser.add_argument('--config', default='.github/project-metrics.json',
                       help='Path to metrics configuration file')
    parser.add_argument('--output', help='Output file for metrics report')
    parser.add_argument('--format', choices=['json', 'markdown'], default='json',
                       help='Output format for report')
    parser.add_argument('--update-config', action='store_true',
                       help='Update the configuration file with collected metrics')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize collector
    collector = MetricsCollector(args.config)
    
    # Collect metrics
    try:
        metrics = collector.collect_all_metrics()
        
        if args.update_config:
            collector.update_config()
        
        # Generate report
        report = collector.generate_report(args.format)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)
        
        print(f"Successfully collected {len(metrics)} metrics")
        
    except Exception as e:
        logging.error(f"Error collecting metrics: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()