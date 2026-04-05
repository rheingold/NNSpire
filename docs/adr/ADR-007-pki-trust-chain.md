# ADR-007 — PKI Certificate Hierarchy for Plugin Signing

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

NNStudio loads third-party plugin shared libraries at runtime. Without authentication,
any malicious actor could distribute a plugin that executes arbitrary code inside the
Studio process, exfiltrates model weights, or corrupts training data.

Three mitigation strategies were evaluated:

| Strategy | Problem |
|---|---|
| Hash allowlist | Requires central server query on every load; offline use breaks; hash list is a single point of failure |
| Code review whitelist | Does not scale; provides a false sense of security for dynamic content |
| **PKI certificate chain** | Standard practice; offline verification; revocable; updatable |

---

## Decision

Plugin authenticity is enforced via a **full PKI certificate hierarchy** with custom X.509 extensions.

```
Root CA  (project owner, HSM-recommended, kept offline)
│
├── Plugin Registry CA (Intermediate, 5-year validity)
│   ├── Community Signing Cert (per open-source author, 2-year, free)
│   ├── Vendor Signing Cert (per commercial author, 1–3-year, paid)
│   └── Trust Update Package Signing Cert (3-year)
│
└── Enterprise Intermediate CA (per qualifying org, PathLenConstraint=0)
    └── Enterprise Plugin Signing Cert (self-issued by enterprise)
```

Custom OIDs are registered under the NNStudio IANA PEN:
- `NNStudioPluginSigning` — EKU for plugin binaries
- `NNStudioTrustUpdate` — EKU for Trust Update Packages
- `NNStudioEnterpriseCA` — EKU marking an Enterprise Intermediate CA

The Root CA certificate is **embedded** in the Studio binary at compile time as a trust seed.
No plugin code executes before `TrustVerifier` has validated the full certificate chain.

**Trust Update Packages (TUPs)** are signed `.tup` bundles that allow the embedded trust
anchors to be updated without a Studio re-install — mandatory from day one to prevent any
future key rotation from stranding installed Studio versions permanently.

---

## Consequences

**Positive**
- Fully offline verification — no network call needed to load a plugin.
- Revocation via CRL / OCSP is supported without mandatory connectivity.
- Enterprise self-signing path reduces friction for large organisations.
- TUP mechanism ensures the trust architecture is forward-compatible across key rotations.

**Negative / constraints**
- Requires the project owner to operate a CA (Root + Intermediate), even if minimal.
- An IANA Private Enterprise Number must be obtained before first public release.
- CRL distribution points and OCSP responder URLs must be high-availability services.
- Plugin development flow requires running `nnstudio-sign` CLI to produce signed artefacts.

**Follow-on**
- Obtain a real IANA PEN before release (placeholder used during development).
- `TrustVerifier`, `TrustStore`, `TrustUpdateHandler` implemented in `nnstudio/plugin-api/`.
- Trust store location: `<app_data>/nnstudio/truststore/` — see ADR-017.
- See TRUST-ARCHITECTURE.md for the complete specification.
