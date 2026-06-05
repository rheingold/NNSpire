# Architecture Decision Records — NNSpire

This directory contains Architecture Decision Records (ADRs) for the NNSpire project.
Each ADR documents a significant architectural choice made during the design and development of NNSpire.

## Format

Each ADR follows the [Michael Nygard template](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions):

- **Title** — short noun phrase naming the decision
- **Status** — `Proposed` · `Accepted` · `Deprecated` · `Superseded by ADR-NNN`
- **Context** — situation that forced the decision
- **Decision** — what was decided
- **Consequences** — resulting trade-offs, follow-on work, constraints

## Index

| ID | Title | Status |
|---|---|---|
| [ADR-001](ADR-001-qt6-ui-framework.md) | Qt 6 as the GUI framework | Accepted |
| [ADR-002](ADR-002-cpp17-engine-language.md) | C++17 as the core engine language | Accepted |
| [ADR-003](ADR-003-c-abi-plugin-boundary.md) | C ABI at the plugin boundary | Accepted |
| [ADR-004](ADR-004-dual-language-everywhere.md) | Dual-language parity (C++ and Python) for every artefact | Accepted |
| [ADR-005](ADR-005-pybind11-python-bridge.md) | pybind11 as the Python bridge | Accepted |
| [ADR-006](ADR-006-license-split.md) | LGPL v3 core / GPL v3 app license split | Accepted |
| [ADR-007](ADR-007-pki-trust-chain.md) | PKI certificate hierarchy for plugin signing | Accepted |
| [ADR-008](ADR-008-onnx-primary-export.md) | ONNX as the primary model export format | Accepted |
| [ADR-009](ADR-009-nns-nnsr-custom-formats.md) | `.nns` and `.nnsr` custom formats for gaps not covered by ONNX | Accepted |
| [ADR-010](ADR-010-cmake-build-system.md) | CMake 3.21+ as the build system | Accepted |
| [ADR-011](ADR-011-no-exceptions-in-engine.md) | No C++ exceptions in the engine core | Accepted |
| [ADR-012](ADR-012-ibackend-dynamic-dispatch.md) | `IBackend` abstraction with dynamically loaded backend shared libraries | Accepted |
| [ADR-013](ADR-013-compute-graph-autograd.md) | Compute graph DAG for automatic differentiation | Accepted |
| [ADR-014](ADR-014-studio-is-server-client.md) | Studio is an inference-server client, not a server host | Accepted |
| [ADR-015](ADR-015-quantum-qiskit-bridge.md) | Quantum backend via Qiskit Python bridge | Accepted |
| [ADR-016](ADR-016-free-forever-feature-tier.md) | Core features permanently free — no paid tier for learning/visualization/pipeline | Accepted |
| [ADR-017](ADR-017-trust-store-in-app-data.md) | Trust store lives in user app-data directory, never in install directory | Accepted |
| [ADR-018](ADR-018-qml-not-qt-widgets.md) | QML (Qt Quick) for the UI, not Qt Widgets | Accepted |
| [ADR-019](ADR-019-inside-out-phase-order.md) | Inside-out phase ordering: core → plugin → UI → pipeline → deploy → quantum | Accepted |
| [ADR-020](ADR-020-iactivation-functional-contract.md) | `IActivation` functional contract: stateless activation with caller-owned context | Accepted |
| [ADR-021](ADR-021-path-equals-namespace-layout.md) | Path = namespace layout rule and three-tier namespace design | Accepted |
