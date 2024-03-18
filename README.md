# quimb-qaoa

This package implements an optimized version of the Quantum Approximate Optimization Algorithm (QAOA) with the open-source library [Quimb](https://github.com/jcmgray/quimb). The intent behind this package is to allow effortless customization of QAOA, such as implementations of different problems and variants.

## Installation

To install the developpement version (recommended):

```
pip install quimb-qaoa@git+https://github.com/quicophy/quimb-qaoa
```

## Structure

quimb-qaoa is partitionned to allow easy implementation of QAOA extensions. Here are the main useful modules.

- Launcher:

Main class of the package.

- Initialization:

Implementation of initialization methods. Currently supports: random initialization and [TQA initialization](https://quantum-journal.org/papers/q-2021-07-01-491).

- Problem:

Implementation of NP-hard problems. Currently supports: monotone NAE3SAT, monotone 1-in-3SAT, monotone 2SAT.

- Hamiltonian:

Formulation of the problems as Ising models.

- Circuit:

Implementation of QAOA ansatz with the circuit form. Currently supports: Regular QAOA, Grover-Mixer QAOA.

- MPS:

Implementation of QAOA ansatz with the Matrix-Product-State (MPS) form. Currently supports: Regular QAOA, Grover-Mixer QAOA.