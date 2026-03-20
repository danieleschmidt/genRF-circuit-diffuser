"""Circuit topology representation with SPICE-like netlist generation."""

import numpy as np
from typing import List, Dict, Optional


COMPONENT_TYPES = {"R", "C", "L", "V", "GND"}

# Encoding for component types in param vector
TYPE_ENCODING = {"R": 0, "C": 1, "L": 2, "V": 3, "GND": 4}
TYPE_DECODING = {v: k for k, v in TYPE_ENCODING.items()}

# Each component encoded as: [type_id, node1, node2, value] -> 4 floats
PARAMS_PER_COMPONENT = 4
MAX_COMPONENTS = 10  # for fixed-size vector encoding


class CircuitTopology:
    """Graph-based circuit topology supporting R, L, C, V, and GND components."""

    def __init__(self, n_nodes: int = 4):
        self.n_nodes = n_nodes
        self._components: List[Dict] = []
        self._next_id = 0

    def add_component(self, comp_type: str, node1: int, node2: int, value: float) -> int:
        """Add a component and return its id."""
        if comp_type not in COMPONENT_TYPES:
            raise ValueError(f"Unknown component type: {comp_type}. Must be one of {COMPONENT_TYPES}")
        if node1 < 0 or node1 >= self.n_nodes:
            raise ValueError(f"node1={node1} out of range [0, {self.n_nodes})")
        if node2 < 0 or node2 >= self.n_nodes:
            raise ValueError(f"node2={node2} out of range [0, {self.n_nodes})")

        comp_id = self._next_id
        self._next_id += 1
        self._components.append({
            "id": comp_id,
            "type": comp_type,
            "node1": node1,
            "node2": node2,
            "value": float(value),
        })
        return comp_id

    def get_components(self) -> List[Dict]:
        """Return list of all components."""
        return list(self._components)

    def get_netlist(self) -> str:
        """Generate SPICE-like netlist string."""
        lines = ["* genRF Circuit Netlist"]
        for comp in self._components:
            t = comp["type"]
            n1 = comp["node1"]
            n2 = comp["node2"]
            v = comp["value"]
            cid = comp["id"]
            if t == "R":
                lines.append(f"R{cid} {n1} {n2} {v:.6g}")
            elif t == "C":
                lines.append(f"C{cid} {n1} {n2} {v:.6g}")
            elif t == "L":
                lines.append(f"L{cid} {n1} {n2} {v:.6g}")
            elif t == "V":
                lines.append(f"V{cid} {n1} {n2} DC {v:.6g}")
            elif t == "GND":
                lines.append(f"* GND{cid} at node {n1}")
        lines.append(".END")
        return "\n".join(lines)

    def adjacency_matrix(self) -> np.ndarray:
        """Return weighted adjacency matrix (sum of component values per node pair)."""
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        for comp in self._components:
            n1, n2 = comp["node1"], comp["node2"]
            if n1 != n2:
                adj[n1, n2] += comp["value"]
                adj[n2, n1] += comp["value"]
        return adj

    def component_vector(self) -> np.ndarray:
        """Flatten circuit parameters to a fixed-size numpy vector for diffusion."""
        vec = np.zeros(MAX_COMPONENTS * PARAMS_PER_COMPONENT, dtype=np.float64)
        for i, comp in enumerate(self._components[:MAX_COMPONENTS]):
            offset = i * PARAMS_PER_COMPONENT
            vec[offset] = TYPE_ENCODING.get(comp["type"], 0)
            vec[offset + 1] = comp["node1"]
            vec[offset + 2] = comp["node2"]
            vec[offset + 3] = comp["value"]
        return vec

    @classmethod
    def from_vector(cls, vec: np.ndarray, n_nodes: int = 4) -> "CircuitTopology":
        """Reconstruct a CircuitTopology from a parameter vector."""
        topo = cls(n_nodes=n_nodes)
        n_components = len(vec) // PARAMS_PER_COMPONENT
        for i in range(n_components):
            offset = i * PARAMS_PER_COMPONENT
            type_id = int(round(float(vec[offset]))) % len(TYPE_DECODING)
            comp_type = TYPE_DECODING.get(type_id, "R")
            node1 = int(abs(round(float(vec[offset + 1])))) % n_nodes
            node2 = int(abs(round(float(vec[offset + 2])))) % n_nodes
            value = abs(float(vec[offset + 3]))
            if value == 0.0:
                value = 1.0  # avoid zero-value components
            topo.add_component(comp_type, node1, node2, value)
        return topo

    def validate(self) -> bool:
        """Check that all nodes are connected and at least one voltage source exists."""
        if not self._components:
            return False

        # Check at least one source (V)
        has_source = any(c["type"] == "V" for c in self._components)
        if not has_source:
            return False

        # Check all nodes are reachable (graph connectivity)
        adj = [set() for _ in range(self.n_nodes)]
        for comp in self._components:
            n1, n2 = comp["node1"], comp["node2"]
            if n1 != n2:
                adj[n1].add(n2)
                adj[n2].add(n1)

        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adj[node] - visited)

        return len(visited) == self.n_nodes
