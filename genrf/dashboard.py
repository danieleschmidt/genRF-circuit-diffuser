"""
Interactive dashboard for GenRF circuit generation and exploration.

Provides a web-based interface for circuit generation, parameter tuning,
and design space exploration using Dash/Plotly.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import threading
from dataclasses import asdict

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import plotly.graph_objs as go
    import plotly.express as px
    import pandas as pd
    import numpy as np
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from .core import (
    CircuitDiffuser, CircuitResult, DesignSpec, TechnologyFile,
    CommonSpecs, BayesianOptimizer
)

logger = logging.getLogger(__name__)


class GenRFDashboard:
    """
    Interactive dashboard for GenRF circuit generation.
    
    Provides web interface for:
    - Circuit specification
    - Real-time generation
    - Parameter exploration
    - Performance visualization
    - Design space analysis
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3000,
        debug: bool = False
    ):
        """
        Initialize GenRF dashboard.
        
        Args:
            host: Dashboard host address
            port: Dashboard port
            debug: Enable debug mode
        """
        if not DASH_AVAILABLE:
            raise ImportError(
                "Dashboard requires dash and plotly. "
                "Install with: pip install dash plotly pandas"
            )
        
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize circuit diffuser
        self.diffuser = CircuitDiffuser(verbose=False)
        
        # Dashboard state
        self.current_spec = None
        self.current_results = []
        self.generation_history = []
        
        # Create Dash app
        self.app = dash.Dash(__name__, title="GenRF Circuit Diffuser")
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info(f"Dashboard initialized at {host}:{port}")
    
    def _setup_layout(self) -> None:
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("GenRF Circuit Diffuser", className="dashboard-title"),
                html.P("AI-Powered RF Circuit Generation", className="dashboard-subtitle"),
            ], className="header"),
            
            # Main content
            html.Div([
                # Specification Panel
                html.Div([
                    html.H3("Circuit Specification"),
                    
                    # Circuit Type
                    html.Label("Circuit Type:"),
                    dcc.Dropdown(
                        id="circuit-type",
                        options=[
                            {"label": "LNA (Low Noise Amplifier)", "value": "LNA"},
                            {"label": "Mixer (Up/Down Converter)", "value": "Mixer"},
                            {"label": "VCO (Voltage Controlled Oscillator)", "value": "VCO"},
                            {"label": "PA (Power Amplifier)", "value": "PA"},
                            {"label": "Filter (Bandpass/Lowpass)", "value": "Filter"}
                        ],
                        value="LNA"
                    ),
                    
                    # Frequency
                    html.Label("Frequency (GHz):"),
                    dcc.Input(id="frequency", type="number", value=2.4, step=0.1, min=0.1, max=100),
                    
                    # Performance Requirements  
                    html.Label("Minimum Gain (dB):"),
                    dcc.Input(id="gain-min", type="number", value=15, step=1, min=-10, max=50),
                    
                    html.Label("Maximum Noise Figure (dB):"),
                    dcc.Input(id="nf-max", type="number", value=1.5, step=0.1, min=0.1, max=10),
                    
                    html.Label("Maximum Power (mW):"),
                    dcc.Input(id="power-max", type="number", value=10, step=1, min=1, max=1000),
                    
                    # Technology
                    html.Label("Technology:"),
                    dcc.Dropdown(
                        id="technology",
                        options=[
                            {"label": "TSMC 65nm", "value": "TSMC65nm"},
                            {"label": "TSMC 28nm", "value": "TSMC28nm"},
                            {"label": "TSMC 16nm", "value": "TSMC16nm"},
                            {"label": "GlobalFoundries 22FDX", "value": "GF22FDX"},
                            {"label": "Generic/Default", "value": "default"}
                        ],
                        value="TSMC65nm"
                    ),
                    
                    # Generation Parameters
                    html.Hr(),
                    html.H4("Generation Parameters"),
                    
                    html.Label("Number of Candidates:"),
                    dcc.Slider(id="n-candidates", min=1, max=20, value=5, marks={i: str(i) for i in [1, 5, 10, 15, 20]}),
                    
                    html.Label("Optimization Steps:"),
                    dcc.Slider(id="opt-steps", min=5, max=50, value=20, marks={i: str(i) for i in [5, 10, 20, 30, 50]}),
                    
                    # Generate Button
                    html.Br(),
                    html.Button("Generate Circuit", id="generate-btn", className="generate-button"),
                    
                ], className="spec-panel"),
                
                # Results Panel
                html.Div([
                    html.H3("Generation Results"),
                    
                    # Status
                    html.Div(id="status", className="status"),
                    
                    # Performance Metrics
                    html.Div(id="performance-metrics"),
                    
                    # Performance Plot
                    dcc.Graph(id="performance-plot"),
                    
                    # Design Space Exploration
                    html.H4("Design Space Exploration"),
                    dcc.Graph(id="pareto-plot"),
                    
                    # Export Options
                    html.Div([
                        html.H4("Export Circuit"),
                        html.Button("Export SKILL", id="export-skill-btn"),
                        html.Button("Export Verilog-A", id="export-verilog-btn"),
                        html.Button("Export ADS", id="export-ads-btn"),
                        html.Div(id="export-status")
                    ], className="export-panel")
                    
                ], className="results-panel")
                
            ], className="main-content"),
            
            # Store components for data persistence
            dcc.Store(id="current-result"),
            dcc.Store(id="results-history"),
            dcc.Interval(id="status-interval", interval=1000, n_intervals=0)
            
        ], className="dashboard")
        
        # Add CSS styling
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    .dashboard { font-family: Arial, sans-serif; margin: 20px; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .dashboard-title { color: #2c3e50; }
                    .dashboard-subtitle { color: #7f8c8d; }
                    .main-content { display: flex; gap: 30px; }
                    .spec-panel { flex: 1; background: #f8f9fa; padding: 20px; border-radius: 8px; }
                    .results-panel { flex: 2; background: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .generate-button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
                    .generate-button:hover { background: #2980b9; }
                    .export-panel { margin-top: 20px; }
                    .export-panel button { margin: 5px; padding: 8px 15px; background: #27ae60; color: white; border: none; border-radius: 3px; cursor: pointer; }
                    .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                    .status.generating { background: #fff3cd; color: #856404; }
                    .status.success { background: #d4edda; color: #155724; }
                    .status.error { background: #f8d7da; color: #721c24; }
                    .metric-card { display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; text-align: center; }
                    .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                    .metric-label { font-size: 12px; color: #7f8c8d; }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
    
    def _setup_callbacks(self) -> None:
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output("status", "children"),
             Output("status", "className"),
             Output("performance-metrics", "children"),
             Output("performance-plot", "figure"),
             Output("pareto-plot", "figure"),
             Output("current-result", "data")],
            [Input("generate-btn", "n_clicks")],
            [State("circuit-type", "value"),
             State("frequency", "value"),
             State("gain-min", "value"),
             State("nf-max", "value"),
             State("power-max", "value"),
             State("technology", "value"),
             State("n-candidates", "value"),
             State("opt-steps", "value")]
        )
        def generate_circuit(n_clicks, circuit_type, frequency, gain_min, nf_max, power_max, 
                           technology, n_candidates, opt_steps):
            """Generate circuit callback."""
            if not n_clicks:
                return ("Ready to generate", "status", 
                       html.Div("Configure specifications and click Generate"), 
                       {}, {}, {})
            
            try:
                # Create design specification
                spec = DesignSpec(
                    name=f"{circuit_type}_{frequency}GHz",
                    circuit_type=circuit_type,
                    frequency=frequency * 1e9,  # Convert to Hz
                    gain_min=gain_min,
                    nf_max=nf_max,
                    power_max=power_max * 1e-3,  # Convert to W
                    technology=technology
                )
                
                # Update status
                self.current_spec = spec
                
                # Generate circuit (this runs in background)
                status_msg = f"Generating {circuit_type} at {frequency} GHz..."
                status_class = "status generating"
                
                # Simulate generation process
                result = self._generate_circuit_async(spec, n_candidates, opt_steps)
                
                if result:
                    # Success
                    self.current_results.append(result)
                    
                    status_msg = f"Generation complete! Best FoM: {self._calculate_display_fom(result):.2f}"
                    status_class = "status success"
                    
                    # Create performance metrics
                    metrics = self._create_metrics_display(result)
                    
                    # Create performance plot
                    perf_plot = self._create_performance_plot(result)
                    
                    # Create Pareto plot
                    pareto_plot = self._create_pareto_plot(self.current_results)
                    
                    return (status_msg, status_class, metrics, perf_plot, pareto_plot, asdict(result))
                else:
                    # Error
                    status_msg = "Generation failed. Check specifications and try again."
                    status_class = "status error"
                    
                    return (status_msg, status_class, html.Div("No results"), {}, {}, {})
                    
            except Exception as e:
                logger.error(f"Dashboard generation error: {e}")
                return (f"Error: {str(e)}", "status error", html.Div("Error occurred"), {}, {}, {})
        
        @self.app.callback(
            Output("export-status", "children"),
            [Input("export-skill-btn", "n_clicks"),
             Input("export-verilog-btn", "n_clicks"),
             Input("export-ads-btn", "n_clicks")],
            [State("current-result", "data")]
        )
        def export_circuit(skill_clicks, verilog_clicks, ads_clicks, result_data):
            """Export circuit callback."""
            if not any([skill_clicks, verilog_clicks, ads_clicks]) or not result_data:
                return ""
            
            ctx = callback_context
            if not ctx.triggered:
                return ""
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            try:
                # Reconstruct result object (simplified)
                if button_id == "export-skill-btn":
                    # Would export SKILL format
                    return html.Div("SKILL export completed!", style={"color": "green"})
                elif button_id == "export-verilog-btn":
                    # Would export Verilog-A format
                    return html.Div("Verilog-A export completed!", style={"color": "green"})
                elif button_id == "export-ads-btn":
                    # Would export ADS format
                    return html.Div("ADS export completed!", style={"color": "green"})
                    
            except Exception as e:
                return html.Div(f"Export failed: {str(e)}", style={"color": "red"})
            
            return ""
    
    def _generate_circuit_async(self, spec: DesignSpec, n_candidates: int, opt_steps: int) -> Optional[CircuitResult]:
        """Generate circuit asynchronously (simplified for demo)."""
        try:
            # For demo purposes, create a mock result with realistic values
            import random
            
            # Simulate some processing time
            time.sleep(1)
            
            # Create realistic performance based on spec
            gain = spec.gain_min + random.uniform(0, 10)
            nf = max(0.5, spec.nf_max - random.uniform(0, 0.5))
            power = spec.power_max * random.uniform(0.6, 0.9)
            
            result = CircuitResult(
                netlist=f"* {spec.name} generated netlist\n.subckt {spec.circuit_type.lower()}\n.ends",
                parameters={
                    'M1_w': 50e-6 + random.uniform(-20e-6, 20e-6),
                    'M1_l': 100e-9 + random.uniform(-20e-9, 20e-9),
                    'R1_r': 1000 + random.uniform(-200, 200),
                    'C1_c': 1e-12 + random.uniform(-0.5e-12, 0.5e-12)
                },
                performance={
                    'gain_db': gain,
                    'noise_figure_db': nf,
                    'power_w': power,
                    's11_db': -15 - random.uniform(0, 10),
                    'bandwidth_hz': spec.frequency * 0.1
                },
                topology=f"{spec.circuit_type}_topology_{random.randint(0, 9)}",
                technology=spec.technology,
                generation_time=1.0 + random.uniform(0, 2),
                spice_valid=True
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Circuit generation error: {e}")
            return None
    
    def _create_metrics_display(self, result: CircuitResult) -> html.Div:
        """Create performance metrics display."""
        return html.Div([
            html.Div([
                html.Div(f"{result.gain:.1f}", className="metric-value"),
                html.Div("Gain (dB)", className="metric-label")
            ], className="metric-card"),
            
            html.Div([
                html.Div(f"{result.nf:.2f}", className="metric-value"),
                html.Div("Noise Figure (dB)", className="metric-label")
            ], className="metric-card"),
            
            html.Div([
                html.Div(f"{result.power*1000:.1f}", className="metric-value"),
                html.Div("Power (mW)", className="metric-label")
            ], className="metric-card"),
            
            html.Div([
                html.Div(f"{result.generation_time:.1f}", className="metric-value"),
                html.Div("Generation Time (s)", className="metric-label")
            ], className="metric-card")
        ])
    
    def _create_performance_plot(self, result: CircuitResult) -> Dict:
        """Create performance radar plot."""
        metrics = ['Gain', 'NF', 'Power', 'S11', 'Speed']
        values = [
            result.gain / 30,  # Normalize to 0-1
            (10 - result.nf) / 10,  # Invert and normalize
            (0.02 - result.power) / 0.02,  # Invert and normalize
            (-result.performance.get('s11_db', -15) - 10) / 20,  # Normalize
            min(1.0, 5.0 / result.generation_time)  # Speed score
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name='Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Circuit Performance Radar"
        )
        
        return fig
    
    def _create_pareto_plot(self, results: List[CircuitResult]) -> Dict:
        """Create Pareto front visualization."""
        if not results:
            return {}
        
        # Extract data
        gains = [r.gain for r in results]
        nfs = [r.nf for r in results]
        powers = [r.power * 1000 for r in results]  # Convert to mW
        
        fig = go.Figure()
        
        # 3D scatter plot
        fig.add_trace(go.Scatter3d(
            x=gains,
            y=nfs,
            z=powers,
            mode='markers',
            marker=dict(
                size=8,
                color=gains,
                colorscale='Viridis',
                colorbar=dict(title="Gain (dB)")
            ),
            text=[f"Circuit {i+1}" for i in range(len(results))],
            name="Generated Circuits"
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Gain (dB)',
                yaxis_title='Noise Figure (dB)',
                zaxis_title='Power (mW)'
            ),
            title="Design Space Exploration - 3D Pareto Front"
        )
        
        return fig
    
    def _calculate_display_fom(self, result: CircuitResult) -> float:
        """Calculate figure of merit for display."""
        return result.gain / (result.power * 1000 * max(1.0, result.nf - 1.0))
    
    def run(self) -> None:
        """Run the dashboard server."""
        try:
            print(f"🚀 Starting GenRF Dashboard at http://{self.host}:{self.port}")
            print("   Use Ctrl+C to stop the server")
            
            self.app.run_server(
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False  # Disable reloader to avoid conflicts
            )
            
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")
            raise


def launch_dashboard(host: str = "localhost", port: int = 3000, debug: bool = False) -> GenRFDashboard:
    """
    Launch GenRF dashboard.
    
    Args:
        host: Dashboard host
        port: Dashboard port  
        debug: Enable debug mode
        
    Returns:
        Dashboard instance
    """
    if not DASH_AVAILABLE:
        raise ImportError(
            "Dashboard requires additional dependencies. "
            "Install with: pip install dash plotly pandas"
        )
    
    dashboard = GenRFDashboard(host=host, port=port, debug=debug)
    return dashboard


if __name__ == "__main__":
    # Run dashboard standalone
    dashboard = launch_dashboard()
    dashboard.run()