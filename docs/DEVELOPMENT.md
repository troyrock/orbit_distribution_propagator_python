# Development checkpoints

1. Completed: audit the Python DSST port and establish configuration equivalence.
2. Completed: sampling and native adaptive propagation with Java/C++ oracles.
3. Completed: bounded process execution, nonlinear statistics and offline viewer.
4. Completed: 62 tests, twenty paired C++ scenarios, browser and installed-wheel gates.
5. Completed: measured worker scaling, full 5,000-particle demo and user/developer docs.

The external Python dependency starts at clean revision
`48001696aea61b4a5629f42ef509c0c8f0d83c87`. The source C++ application is at
`d88bb04`; its source and dependency were not modified by this port.

Development commits separate package scaffolding, native backend/sampling,
parallel simulation/validation, and final documentation. Reproduce validation
with the commands in README. Generated scientific results and benchmark logs
are ignored; compact measured evidence is tracked in docs/validation_results.json.
