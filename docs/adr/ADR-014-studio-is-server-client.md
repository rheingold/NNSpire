# ADR-014 — Studio is an Inference-Server Client, Not a Server Host

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

NNStudio needs to support running trained models against real inference backends
(Triton Inference Server, TF Serving, KServe, ONNX Runtime, OpenAI-compatible REST, etc.)
as part of its deployment testing and pipeline execution features.

Two architectural choices exist:

| Option | Description |
|---|---|
| **Bundle a server** | Ship Triton/ONNX Runtime as a subprocess inside the Studio; manage its lifecycle |
| **Client only** | Studio connects to externally operated servers via standard protocols; never owns the server process |

Bundling a server would mean:
- Shipping large server binaries (Triton image is several GB).
- Managing subprocess lifecycle, port allocation, and crash recovery in the Studio.
- Binding to a specific server version and its upgrade cycle.
- Defeating the purpose of testing against real production infrastructure.

---

## Decision

**NNStudio is a client of inference servers. It does not bundle, embed, or own any inference server.**

- Each supported server is accessed via a **runner connector** plugin implementing `IRunnerClient`.
- Supported connectors (built-in): Triton gRPC v2, TF Serving REST, KServe REST,
  ONNX Runtime embedded (in-process, the one exception — a lightweight local runner for
  development/testing only), OpenAI-compatible REST.
- As a **development convenience only**, the Studio may assist launching a local Docker
  container for a supported server. It does not own the container lifecycle beyond
  start/stop; the container is externally managed once started.
- The `.nnsr` runner bundle (ADR-009) is self-contained for deployment purposes
  and does not depend on the Studio being present at inference time.

All runner connectors ship in both C++ and Python forms (ADR-004).

---

## Consequences

**Positive**
- Studio remains lightweight — no multi-GB server binaries.
- Users test against the same server software they use in production.
- New server types can be added as runner connector plugins without modifying the Studio.
- Connector protocol is standard (gRPC v2 / OpenAPI) — no proprietary lock-in.

**Negative / constraints**
- Users must have access to a running inference server for full deployment testing.
- ONNX Runtime embedded connector blurs the line slightly — but it is intentionally
  scoped to in-process development use only and documented as such.
- Docker convenience launcher adds a Docker dependency for that workflow only.

**Follow-on**
- `IRunnerClient` interface in `nnstudio/deployment/`.
- Built-in connectors in `nnstudio/deployment/connectors/`.
- Docker launcher helper in `nnstudio/deployment/docker_launcher/` (Phase 5).
- See DEPLOYMENT.md §3 for the full runner connector specification.
