# Distribution propagator — Python

Native Python equivalent of the sibling C++ Monte Carlo uncertainty simulator.
It uses the independent Python Orekit DSST port, with the same configurations,
metrics, outputs and offline HTML visualization. The C++ application is used
only as an optional validation oracle, never as a propagation backend.

Implementation and validation are in progress. Runtime requirements are Python
3.10+, NumPy, and the external native `dsst-python` package. No Java runtime or
C++ compiler is required to run the Python application.
