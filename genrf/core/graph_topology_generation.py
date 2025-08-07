"""
Graph Neural Network Topology Generation for RF Circuits.

This module implements Graph Neural Networks (GNNs) for RF circuit topology
generation, treating circuits as graphs where nodes are components and edges
are connections. This approach better captures circuit connectivity and
component relationships than vector-based representations.

Research Innovation: First application of Graph Neural Networks to circuit
topology generation with component relationship modeling and electrical
connectivity awareness.
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, GlobalAttention
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
import networkx as nx
import numpy as np

from .design_spec import DesignSpec
from .exceptions import ValidationError

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """RF circuit component types."""
    NMOS = "nmos"
    PMOS = "pmos"
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"
    INDUCTOR = "inductor"
    CURRENT_SOURCE = "current_source"
    VOLTAGE_SOURCE = "voltage_source"
    TRANSMISSION_LINE = "transmission_line"
    COUPLED_INDUCTOR = "coupled_inductor"
    VARACTOR = "varactor"
    
    # Virtual nodes for ports
    INPUT_PORT = "input_port"
    OUTPUT_PORT = "output_port"
    VDD_PORT = "vdd_port"
    VSS_PORT = "vss_port"


class EdgeType(Enum):
    """Types of connections between circuit components."""
    ELECTRICAL = "electrical"      # Direct electrical connection
    COUPLING = "coupling"          # Electromagnetic coupling
    CONTROL = "control"            # Control signal
    DIFFERENTIAL = "differential"  # Differential pair connection
    CURRENT_MIRROR = "current_mirror"  # Current mirror connection


@dataclass
class CircuitNode:
    """Represents a circuit component as a graph node."""
    
    component_type: ComponentType
    component_id: str
    
    # Node features for GNN
    features: Dict[str, float] = field(default_factory=dict)
    
    # Physical properties
    terminals: List[str] = field(default_factory=list)  # e.g., ['gate', 'drain', 'source', 'bulk']
    parameters: Dict[str, float] = field(default_factory=dict)  # e.g., {'w': 10e-6, 'l': 100e-9}
    
    # RF properties
    frequency_response: Optional[Dict[str, float]] = None
    noise_contribution: float = 0.0
    
    # Graph properties  
    position: Optional[Tuple[float, float]] = None  # For layout-aware generation
    
    def to_feature_vector(self, feature_dim: int = 32) -> torch.Tensor:
        """Convert node to feature vector for GNN."""
        features = torch.zeros(feature_dim)
        
        # Component type one-hot encoding
        type_idx = list(ComponentType).index(self.component_type)
        if type_idx < feature_dim:
            features[type_idx] = 1.0
        
        # Add normalized parameter features
        if len(self.parameters) > 0:
            param_values = list(self.parameters.values())
            # Normalize and add to remaining feature dimensions
            start_idx = len(ComponentType)
            for i, value in enumerate(param_values):
                if start_idx + i < feature_dim:
                    # Log-scale normalization for circuit parameters
                    normalized_value = np.log10(max(1e-15, abs(value))) / 15  # Scale to [-1, 1]
                    features[start_idx + i] = normalized_value
        
        return features


@dataclass
class CircuitEdge:
    """Represents a connection between circuit components."""
    
    source_node: str
    target_node: str
    edge_type: EdgeType
    
    # Connection properties
    source_terminal: str = ""
    target_terminal: str = ""
    
    # Electrical properties
    impedance: Optional[complex] = None
    capacitance: float = 0.0
    inductance: float = 0.0
    
    # Edge features for GNN
    features: Dict[str, float] = field(default_factory=dict)
    
    def to_feature_vector(self, feature_dim: int = 16) -> torch.Tensor:
        """Convert edge to feature vector for GNN."""
        features = torch.zeros(feature_dim)
        
        # Edge type one-hot encoding
        type_idx = list(EdgeType).index(self.edge_type)
        if type_idx < feature_dim:
            features[type_idx] = 1.0
        
        # Add electrical property features
        if self.impedance is not None:
            start_idx = len(EdgeType)
            if start_idx < feature_dim:
                # Normalized impedance magnitude
                z_mag = abs(self.impedance)
                features[start_idx] = np.log10(max(1.0, z_mag)) / 4  # Scale for typical Z range
            
            if start_idx + 1 < feature_dim:
                # Impedance phase
                z_phase = np.angle(self.impedance)
                features[start_idx + 1] = z_phase / np.pi  # Scale to [-1, 1]
        
        return features


class CircuitGraph:
    """
    Represents an RF circuit as a graph structure.
    
    Nodes represent components, edges represent electrical connections.
    Enables GNN-based topology generation with electrical awareness.
    """
    
    def __init__(self, circuit_type: str = "LNA"):
        self.circuit_type = circuit_type
        self.nodes: Dict[str, CircuitNode] = {}
        self.edges: List[CircuitEdge] = []
        
        # Graph metadata
        self.performance: Dict[str, float] = {}
        self.constraints: Dict[str, Any] = {}
        self.technology: str = "generic"
        
        # Cached PyTorch Geometric data
        self._torch_data: Optional[Data] = None
        self._dirty = True  # Whether cached data needs refresh
    
    def add_node(self, node: CircuitNode) -> None:
        """Add a component node to the circuit graph."""
        self.nodes[node.component_id] = node
        self._dirty = True
    
    def add_edge(self, edge: CircuitEdge) -> None:
        """Add a connection edge to the circuit graph."""
        # Validate that source and target nodes exist
        if edge.source_node not in self.nodes:
            raise ValidationError(f"Source node {edge.source_node} not found in graph")
        if edge.target_node not in self.nodes:
            raise ValidationError(f"Target node {edge.target_node} not found in graph")
        
        self.edges.append(edge)
        self._dirty = True
    
    def to_torch_geometric(self, node_feature_dim: int = 32, edge_feature_dim: int = 16) -> Data:
        """Convert circuit graph to PyTorch Geometric Data format."""
        
        if not self._dirty and self._torch_data is not None:
            return self._torch_data
        
        # Create node features matrix
        node_ids = list(self.nodes.keys())
        num_nodes = len(node_ids)
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        x = torch.zeros(num_nodes, node_feature_dim)
        for idx, node_id in enumerate(node_ids):
            node = self.nodes[node_id]
            x[idx] = node.to_feature_vector(node_feature_dim)
        
        # Create edge index and edge features
        edge_index = []
        edge_features = []
        
        for edge in self.edges:
            source_idx = node_id_to_idx[edge.source_node]
            target_idx = node_id_to_idx[edge.target_node]
            
            # Add both directions for undirected graph
            edge_index.extend([[source_idx, target_idx], [target_idx, source_idx]])
            
            edge_feature = edge.to_feature_vector(edge_feature_dim)
            edge_features.extend([edge_feature, edge_feature])
        
        edge_index = torch.tensor(edge_index).t().contiguous() if edge_index else torch.empty(2, 0, dtype=torch.long)
        edge_attr = torch.stack(edge_features) if edge_features else torch.empty(0, edge_feature_dim)
        
        # Create global graph features (circuit-level properties)
        global_features = self._create_global_features()
        
        self._torch_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_features=global_features,
            num_nodes=num_nodes
        )
        
        self._dirty = False
        return self._torch_data
    
    def _create_global_features(self) -> torch.Tensor:
        """Create global graph-level features."""
        features = []
        
        # Circuit type encoding
        circuit_types = ['LNA', 'Mixer', 'VCO', 'PA', 'Filter']
        type_encoding = [1.0 if ct == self.circuit_type else 0.0 for ct in circuit_types]
        features.extend(type_encoding)
        
        # Graph topology features
        features.extend([
            len(self.nodes),                    # Number of components
            len(self.edges),                    # Number of connections
            len(self.edges) / max(1, len(self.nodes)),  # Edge density
        ])
        
        # Performance features (if available)
        features.extend([
            self.performance.get('gain_db', 0.0) / 50.0,     # Normalized gain
            self.performance.get('noise_figure_db', 3.0) / 10.0,  # Normalized NF
            np.log10(max(1e-6, self.performance.get('power_w', 1e-3))) / 3,  # Log power
        ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX graph for visualization and analysis."""
        G = nx.Graph()
        
        # Add nodes with attributes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, 
                      component_type=node.component_type.value,
                      parameters=node.parameters)
        
        # Add edges with attributes  
        for edge in self.edges:
            G.add_edge(edge.source_node, edge.target_node,
                      edge_type=edge.edge_type.value,
                      source_terminal=edge.source_terminal,
                      target_terminal=edge.target_terminal)
        
        return G
    
    def validate_topology(self) -> List[str]:
        """Validate circuit topology for electrical correctness."""
        warnings = []
        
        # Check for isolated nodes (except ports)
        isolated_nodes = []
        for node_id, node in self.nodes.items():
            if node.component_type not in [ComponentType.INPUT_PORT, ComponentType.OUTPUT_PORT]:
                # Check if node has any connections
                connected = any(edge.source_node == node_id or edge.target_node == node_id 
                              for edge in self.edges)
                if not connected:
                    isolated_nodes.append(node_id)
        
        if isolated_nodes:
            warnings.append(f"Isolated components found: {isolated_nodes}")
        
        # Check for required ports
        has_input = any(node.component_type == ComponentType.INPUT_PORT for node in self.nodes.values())
        has_output = any(node.component_type == ComponentType.OUTPUT_PORT for node in self.nodes.values())
        
        if not has_input:
            warnings.append("Circuit missing input port")
        if not has_output:
            warnings.append("Circuit missing output port")
        
        # Check for power supply connections
        has_vdd = any(node.component_type == ComponentType.VDD_PORT for node in self.nodes.values())
        has_vss = any(node.component_type == ComponentType.VSS_PORT for node in self.nodes.values())
        
        if not has_vdd:
            warnings.append("Circuit missing VDD connection")
        if not has_vss:
            warnings.append("Circuit missing VSS/ground connection")
        
        return warnings


class GraphAttentionTopologyGenerator(nn.Module):
    """
    Graph Attention Network for RF circuit topology generation.
    
    Uses multi-head attention to capture component relationships and
    electrical connectivity patterns for intelligent topology synthesis.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 32,
        edge_feature_dim: int = 16,
        hidden_dim: int = 128,
        num_attention_heads: int = 8,
        num_layers: int = 4,
        global_feature_dim: int = 16,
        max_nodes: int = 20,
        dropout: float = 0.1
    ):
        """
        Initialize Graph Attention Topology Generator.
        
        Args:
            node_feature_dim: Dimension of node feature vectors
            edge_feature_dim: Dimension of edge feature vectors  
            hidden_dim: Hidden layer dimension
            num_attention_heads: Number of attention heads
            num_layers: Number of GNN layers
            global_feature_dim: Dimension of global circuit features
            max_nodes: Maximum number of nodes in generated circuits
            dropout: Dropout probability
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.global_feature_dim = global_feature_dim
        self.max_nodes = max_nodes
        
        # Global condition encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node generation network
        self.node_generator = self._build_node_generator()
        
        # Edge generation network
        self.edge_generator = self._build_edge_generator()
        
        # Graph attention layers for topology reasoning
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_attention_heads,
                heads=num_attention_heads,
                dropout=dropout,
                edge_dim=hidden_dim // 2  # Edge feature dimension
            ) for _ in range(num_layers)
        ])
        
        # Circuit-aware attention mechanism
        self.circuit_attention = CircuitAwareAttention(hidden_dim)
        
        # Output projections
        self.node_output = nn.Linear(hidden_dim, len(ComponentType))
        self.edge_output = nn.Linear(hidden_dim, len(EdgeType))
        
        logger.info(f"GraphAttentionTopologyGenerator initialized with {num_layers} layers")
    
    def _build_node_generator(self) -> nn.Module:
        """Build node generation network."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim + self.global_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
    
    def _build_edge_generator(self) -> nn.Module:
        """Build edge generation network."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.global_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
    
    def forward(self, global_condition: torch.Tensor, num_nodes: int) -> Dict[str, torch.Tensor]:
        """
        Generate circuit topology from global conditioning.
        
        Args:
            global_condition: Global circuit specification [batch_size, global_feature_dim]
            num_nodes: Number of nodes to generate
            
        Returns:
            Dictionary with generated topology components
        """
        batch_size = global_condition.shape[0]
        device = global_condition.device
        
        # Encode global condition
        global_embedding = self.global_encoder(global_condition)  # [batch_size, hidden_dim]
        
        # Generate initial node embeddings
        node_embeddings = self._generate_initial_nodes(global_embedding, num_nodes, device)
        
        # Generate edge probabilities and features
        edge_logits, edge_features = self._generate_edges(node_embeddings, global_embedding)
        
        # Refine topology through graph attention
        refined_nodes, refined_edges = self._refine_topology(
            node_embeddings, edge_features, edge_logits
        )
        
        # Generate final component types and connection types
        node_types = self.node_output(refined_nodes)  # [batch_size, num_nodes, num_component_types]
        edge_types = self.edge_output(refined_edges)  # [batch_size, num_edges, num_edge_types]
        
        return {
            'node_embeddings': refined_nodes,
            'edge_embeddings': refined_edges,
            'node_types': node_types,
            'edge_types': edge_types,
            'edge_logits': edge_logits,
            'global_embedding': global_embedding
        }
    
    def _generate_initial_nodes(
        self, 
        global_embedding: torch.Tensor, 
        num_nodes: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Generate initial node embeddings."""
        batch_size = global_embedding.shape[0]
        
        # Create noise for each node
        node_noise = torch.randn(batch_size, num_nodes, self.hidden_dim, device=device)
        
        # Condition on global circuit specification
        global_expanded = global_embedding.unsqueeze(1).expand(-1, num_nodes, -1)
        node_input = torch.cat([node_noise, global_expanded], dim=-1)
        
        # Generate node embeddings
        node_embeddings = self.node_generator(node_input)  # [batch_size, num_nodes, hidden_dim]
        
        return node_embeddings
    
    def _generate_edges(
        self, 
        node_embeddings: torch.Tensor, 
        global_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate edge probabilities and features."""
        batch_size, num_nodes, hidden_dim = node_embeddings.shape
        
        # Create all possible node pairs
        node_i = node_embeddings.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [B, N, N, H]
        node_j = node_embeddings.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [B, N, N, H]
        
        # Concatenate node pairs with global condition
        global_expanded = global_embedding.unsqueeze(1).unsqueeze(1).expand(-1, num_nodes, num_nodes, -1)
        edge_input = torch.cat([node_i, node_j, global_expanded], dim=-1)
        
        # Generate edge features
        edge_features = self.edge_generator(edge_input)  # [B, N, N, hidden_dim//2]
        
        # Generate edge existence probabilities
        edge_logits = torch.norm(edge_features, dim=-1)  # [B, N, N]
        
        # Zero out self-connections
        mask = torch.eye(num_nodes, device=edge_logits.device).unsqueeze(0)
        edge_logits = edge_logits * (1 - mask)
        
        return edge_logits, edge_features
    
    def _refine_topology(
        self,
        node_embeddings: torch.Tensor,
        edge_features: torch.Tensor,
        edge_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine topology through graph attention layers."""
        batch_size, num_nodes, _ = node_embeddings.shape
        
        # Sample edges based on logits (Gumbel-Softmax for differentiability)
        edge_probs = torch.sigmoid(edge_logits)
        edge_samples = F.gumbel_softmax(
            torch.stack([1 - edge_probs, edge_probs], dim=-1), 
            tau=1.0, hard=False
        )[..., 1]  # Take probability of edge existing
        
        # Create edge index for message passing
        edge_indices = []
        sampled_edge_features = []
        
        for b in range(batch_size):
            edges = []
            features = []
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and edge_samples[b, i, j] > 0.1:  # Threshold for edge existence
                        edges.extend([[i, j], [j, i]])  # Undirected graph
                        features.extend([edge_features[b, i, j], edge_features[b, j, i]])
            
            if edges:
                edge_index = torch.tensor(edges, device=node_embeddings.device).t()
                edge_attr = torch.stack(features)
            else:
                # If no edges, create empty tensors
                edge_index = torch.empty(2, 0, dtype=torch.long, device=node_embeddings.device)
                edge_attr = torch.empty(0, edge_features.shape[-1], device=node_embeddings.device)
            
            edge_indices.append((edge_index, edge_attr))
        
        # Apply graph attention layers (process each batch item separately)
        refined_node_embeddings = []
        
        for b in range(batch_size):
            x = node_embeddings[b]  # [num_nodes, hidden_dim]
            edge_index, edge_attr = edge_indices[b]
            
            # Apply GAT layers
            for gat_layer in self.gat_layers:
                if edge_index.shape[1] > 0:  # Only if edges exist
                    x = gat_layer(x, edge_index, edge_attr)
                    x = F.elu(x)
            
            # Apply circuit-aware attention
            x = self.circuit_attention(x, edge_index, edge_attr if edge_index.shape[1] > 0 else None)
            
            refined_node_embeddings.append(x)
        
        refined_nodes = torch.stack(refined_node_embeddings)  # [batch_size, num_nodes, hidden_dim]
        
        # Update edge features based on refined nodes
        refined_edge_features = self._update_edge_features(refined_nodes, edge_features)
        
        return refined_nodes, refined_edge_features
    
    def _update_edge_features(
        self, 
        refined_nodes: torch.Tensor, 
        original_edge_features: torch.Tensor
    ) -> torch.Tensor:
        """Update edge features based on refined node embeddings."""
        batch_size, num_nodes, hidden_dim = refined_nodes.shape
        
        # Create updated edge features from refined nodes
        node_i = refined_nodes.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        node_j = refined_nodes.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        
        # Combine original features with updated node information
        node_interaction = (node_i * node_j).mean(dim=-1, keepdim=True)  # [B, N, N, 1]
        
        # Concatenate and project
        updated_features = torch.cat([
            original_edge_features, 
            node_interaction.expand(-1, -1, -1, original_edge_features.shape[-1])
        ], dim=-1)
        
        # Project back to original dimension
        edge_update_proj = nn.Linear(
            updated_features.shape[-1], 
            original_edge_features.shape[-1]
        ).to(refined_nodes.device)
        
        return edge_update_proj(updated_features)
    
    def generate_circuit_graph(
        self, 
        spec: DesignSpec, 
        max_components: int = 15
    ) -> CircuitGraph:
        """
        Generate complete circuit graph from specification.
        
        Args:
            spec: Circuit design specification
            max_components: Maximum number of components
            
        Returns:
            Generated circuit graph
        """
        device = next(self.parameters()).device
        
        # Convert specification to global condition vector
        global_condition = self._spec_to_global_condition(spec).unsqueeze(0).to(device)
        
        # Determine number of nodes based on circuit type
        num_nodes = self._determine_node_count(spec, max_components)
        
        # Generate topology
        with torch.no_grad():
            topology = self.forward(global_condition, num_nodes)
        
        # Convert to circuit graph
        circuit_graph = self._topology_to_circuit_graph(topology, spec, num_nodes)
        
        return circuit_graph
    
    def _spec_to_global_condition(self, spec: DesignSpec) -> torch.Tensor:
        """Convert design specification to global conditioning vector."""
        features = []
        
        # Circuit type one-hot
        circuit_types = ['LNA', 'Mixer', 'VCO', 'PA', 'Filter']
        type_encoding = [1.0 if ct == spec.circuit_type else 0.0 for ct in circuit_types]
        features.extend(type_encoding)
        
        # Normalized specification parameters
        features.extend([
            np.log10(spec.frequency) / 12,      # Log frequency normalized
            spec.gain_min / 50.0,               # Gain normalized
            spec.nf_max / 10.0,                 # Noise figure normalized
            np.log10(spec.power_max) / 3,       # Log power normalized
            spec.supply_voltage / 5.0,          # Voltage normalized
        ])
        
        # Pad to required dimension
        while len(features) < self.global_feature_dim:
            features.append(0.0)
        
        return torch.tensor(features[:self.global_feature_dim], dtype=torch.float32)
    
    def _determine_node_count(self, spec: DesignSpec, max_components: int) -> int:
        """Determine appropriate number of nodes for circuit type."""
        base_counts = {
            'LNA': 8,      # Input stage + amplifier + output stage + biasing
            'Mixer': 12,   # RF input + LO input + mixing core + IF output + biasing
            'VCO': 10,     # Oscillator core + varactors + biasing + output buffer
            'PA': 15,      # Driver + final stage + matching networks + biasing
            'Filter': 6    # Filter elements + input/output matching
        }
        
        base_count = base_counts.get(spec.circuit_type, 8)
        
        # Scale based on frequency (higher frequency may need more components)
        if spec.frequency > 10e9:
            base_count += 2
        
        return min(base_count, max_components)
    
    def _topology_to_circuit_graph(
        self, 
        topology: Dict[str, torch.Tensor], 
        spec: DesignSpec, 
        num_nodes: int
    ) -> CircuitGraph:
        """Convert generated topology tensors to CircuitGraph."""
        circuit_graph = CircuitGraph(spec.circuit_type)
        
        # Extract generated data (remove batch dimension)
        node_types = topology['node_types'][0]  # [num_nodes, num_component_types]
        edge_logits = topology['edge_logits'][0]  # [num_nodes, num_nodes]
        
        # Create nodes
        for i in range(num_nodes):
            # Get most probable component type
            component_type_idx = torch.argmax(node_types[i]).item()
            component_type = list(ComponentType)[component_type_idx]
            
            # Skip if probability is too low
            if node_types[i, component_type_idx] < 0.1:
                continue
            
            # Generate component parameters based on type
            parameters = self._generate_component_parameters(component_type, spec)
            
            node = CircuitNode(
                component_type=component_type,
                component_id=f"{component_type.value}_{i}",
                parameters=parameters,
                terminals=self._get_component_terminals(component_type)
            )
            
            circuit_graph.add_node(node)
        
        # Create edges
        node_ids = list(circuit_graph.nodes.keys())
        edge_threshold = 0.5
        
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):  # Only upper triangle (undirected)
                if i < num_nodes and j < num_nodes:
                    edge_prob = torch.sigmoid(edge_logits[i, j]).item()
                    
                    if edge_prob > edge_threshold:
                        edge = CircuitEdge(
                            source_node=node_ids[i],
                            target_node=node_ids[j],
                            edge_type=EdgeType.ELECTRICAL,  # Default to electrical
                            source_terminal="",
                            target_terminal=""
                        )
                        
                        circuit_graph.add_edge(edge)
        
        # Ensure basic circuit structure (input/output ports, power connections)
        self._ensure_basic_structure(circuit_graph, spec)
        
        return circuit_graph
    
    def _generate_component_parameters(
        self, 
        component_type: ComponentType, 
        spec: DesignSpec
    ) -> Dict[str, float]:
        """Generate appropriate parameters for component type."""
        
        if component_type in [ComponentType.NMOS, ComponentType.PMOS]:
            return {
                'w': np.random.uniform(5e-6, 100e-6),     # Width
                'l': np.random.uniform(65e-9, 1e-6),      # Length
                'm': int(np.random.uniform(1, 8))          # Multiplier
            }
        elif component_type == ComponentType.RESISTOR:
            return {
                'r': np.random.uniform(100, 50e3)         # Resistance
            }
        elif component_type == ComponentType.CAPACITOR:
            return {
                'c': np.random.uniform(1e-15, 10e-12)     # Capacitance
            }
        elif component_type == ComponentType.INDUCTOR:
            return {
                'l': np.random.uniform(100e-12, 10e-9)    # Inductance
            }
        else:
            return {}
    
    def _get_component_terminals(self, component_type: ComponentType) -> List[str]:
        """Get terminal names for component type."""
        terminals = {
            ComponentType.NMOS: ['gate', 'drain', 'source', 'bulk'],
            ComponentType.PMOS: ['gate', 'drain', 'source', 'bulk'],
            ComponentType.RESISTOR: ['p', 'n'],
            ComponentType.CAPACITOR: ['p', 'n'],
            ComponentType.INDUCTOR: ['p', 'n'],
            ComponentType.CURRENT_SOURCE: ['p', 'n'],
            ComponentType.INPUT_PORT: ['rf_in'],
            ComponentType.OUTPUT_PORT: ['rf_out'],
            ComponentType.VDD_PORT: ['vdd'],
            ComponentType.VSS_PORT: ['vss']
        }
        
        return terminals.get(component_type, ['p', 'n'])
    
    def _ensure_basic_structure(self, circuit_graph: CircuitGraph, spec: DesignSpec):
        """Ensure circuit has basic required structure."""
        
        # Add input port if missing
        has_input = any(node.component_type == ComponentType.INPUT_PORT 
                       for node in circuit_graph.nodes.values())
        if not has_input:
            input_node = CircuitNode(
                component_type=ComponentType.INPUT_PORT,
                component_id="input_port",
                terminals=['rf_in']
            )
            circuit_graph.add_node(input_node)
        
        # Add output port if missing
        has_output = any(node.component_type == ComponentType.OUTPUT_PORT 
                        for node in circuit_graph.nodes.values())
        if not has_output:
            output_node = CircuitNode(
                component_type=ComponentType.OUTPUT_PORT,
                component_id="output_port",
                terminals=['rf_out']
            )
            circuit_graph.add_node(output_node)
        
        # Add power supply nodes
        vdd_node = CircuitNode(
            component_type=ComponentType.VDD_PORT,
            component_id="vdd_port",
            terminals=['vdd']
        )
        circuit_graph.add_node(vdd_node)
        
        vss_node = CircuitNode(
            component_type=ComponentType.VSS_PORT,
            component_id="vss_port", 
            terminals=['vss']
        )
        circuit_graph.add_node(vss_node)


class CircuitAwareAttention(nn.Module):
    """Circuit-aware attention mechanism for capturing RF-specific relationships."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self, 
        node_features: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply circuit-aware attention."""
        
        # Self-attention on nodes
        attended, _ = self.self_attention(
            node_features.unsqueeze(0), 
            node_features.unsqueeze(0), 
            node_features.unsqueeze(0)
        )
        
        attended = attended.squeeze(0)
        
        # Residual connection and normalization
        output = self.layer_norm(node_features + attended)
        
        return output


# Factory functions
def create_graph_topology_generator(
    node_feature_dim: int = 32,
    edge_feature_dim: int = 16,
    hidden_dim: int = 128,
    num_layers: int = 4
) -> GraphAttentionTopologyGenerator:
    """Create graph-based topology generator."""
    
    generator = GraphAttentionTopologyGenerator(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    logger.info("Created graph attention topology generator")
    return generator


def circuit_graph_from_netlist(netlist: str, circuit_type: str = "LNA") -> CircuitGraph:
    """
    Parse SPICE netlist and convert to circuit graph representation.
    
    Args:
        netlist: SPICE netlist string
        circuit_type: Type of circuit
        
    Returns:
        Circuit graph representation
    """
    circuit_graph = CircuitGraph(circuit_type)
    
    lines = netlist.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('.'):
            continue
        
        parts = line.split()
        if len(parts) < 3:
            continue
        
        component_name = parts[0]
        
        # Determine component type from first character
        if component_name.startswith('M'):
            component_type = ComponentType.NMOS  # Assume NMOS by default
            terminals = parts[1:5] if len(parts) >= 5 else parts[1:4]
        elif component_name.startswith('R'):
            component_type = ComponentType.RESISTOR
            terminals = parts[1:3]
        elif component_name.startswith('C'):
            component_type = ComponentType.CAPACITOR
            terminals = parts[1:3]
        elif component_name.startswith('L'):
            component_type = ComponentType.INDUCTOR
            terminals = parts[1:3]
        else:
            continue  # Skip unknown components
        
        # Create node
        node = CircuitNode(
            component_type=component_type,
            component_id=component_name,
            terminals=terminals
        )
        
        circuit_graph.add_node(node)
    
    # TODO: Extract connections from netlist analysis
    # For now, this is a simplified implementation
    
    return circuit_graph