GenRF Circuit Diffuser Documentation
====================================

**GenRF Circuit Diffuser** is a toolkit for generating analog and RF circuits using 
cycle-consistent GANs and diffusion models with SPICE-in-the-loop optimization.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   tutorials/index
   examples/index

Features
--------

* Hybrid generative models combining Cycle-GAN and diffusion approaches
* SPICE-in-the-loop validation for accurate circuit simulation
* Multi-objective optimization for gain, noise figure, and power
* Export to industry-standard EDA tools (Cadence, Keysight)
* Interactive Grafana dashboard for design space exploration

Quick Example
-------------

.. code-block:: python

   from genrf import CircuitDiffuser, DesignSpec

   # Define LNA specifications
   spec = DesignSpec(
       circuit_type="LNA",
       frequency=2.4e9,
       gain_min=15,
       nf_max=1.5,
       power_max=10e-3
   )

   # Generate optimized circuit
   diffuser = CircuitDiffuser()
   circuit = diffuser.generate(spec)
   circuit.export_skill("lna_design.il")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`