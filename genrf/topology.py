"""CircuitTopology: nodes and components (R/L/C/transistor) definition."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Component:
    """A single circuit component."""
    ctype: str          # 'R', 'L', 'C', or 'transistor'
    node_a: str
    node_b: str
    value: float        # Ohms / Henries / Farads; for transistor: gm in S
    name: str = ""
    node_c: Optional[str] = None  # collector/gate for transistor

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.ctype}_{self.node_a}_{self.node_b}"

    @property
    def is_passive(self) -> bool:
        return self.ctype in ("R", "L", "C")


class CircuitTopology:
    """Represent a circuit as nodes and components."""

    def __init__(self, name: str = "Circuit"):
        self.name = name
        self.components: List[Component] = []
        self._node_set: set = {"GND"}

    def add_component(self, ctype: str, node_a: str, node_b: str, value: float,
                      name: str = "", node_c: Optional[str] = None) -> Component:
        """Add a component to the circuit."""
        comp = Component(ctype=ctype, node_a=node_a, node_b=node_b,
                         value=value, name=name, node_c=node_c)
        self.components.append(comp)
        self._node_set.add(node_a)
        self._node_set.add(node_b)
        if node_c:
            self._node_set.add(node_c)
        return comp

    def add_resistor(self, node_a: str, node_b: str, r: float, name: str = "") -> Component:
        return self.add_component("R", node_a, node_b, r, name)

    def add_inductor(self, node_a: str, node_b: str, l: float, name: str = "") -> Component:
        return self.add_component("L", node_a, node_b, l, name)

    def add_capacitor(self, node_a: str, node_b: str, c: float, name: str = "") -> Component:
        return self.add_component("C", node_a, node_b, c, name)

    def add_transistor(self, node_d: str, node_g: str, node_s: str,
                       gm: float, name: str = "") -> Component:
        """Add a transistor model (simplified: transconductance gm in S)."""
        return self.add_component("transistor", node_d, node_s, gm, name, node_c=node_g)

    @property
    def nodes(self) -> List[str]:
        return sorted(self._node_set)

    def get_components_by_type(self, ctype: str) -> List[Component]:
        return [c for c in self.components if c.ctype == ctype]

    def parameter_vector(self) -> List[float]:
        """Flatten component values to a parameter vector."""
        return [c.value for c in self.components]

    def update_from_vector(self, values: List[float]):
        """Update component values from a vector."""
        for i, comp in enumerate(self.components):
            if i < len(values):
                comp.value = max(values[i], 1e-15)  # keep positive

    def __repr__(self):
        return f"CircuitTopology({self.name}, {len(self.components)} components, {len(self.nodes)} nodes)"

    @classmethod
    def lc_filter(cls, L: float = 1e-6, C: float = 1e-12, R_load: float = 50.0) -> "CircuitTopology":
        """Build a simple LC low-pass filter."""
        topo = cls("LC_Filter")
        topo.add_inductor("IN", "OUT", L, "L1")
        topo.add_capacitor("OUT", "GND", C, "C1")
        topo.add_resistor("OUT", "GND", R_load, "R_load")
        return topo
