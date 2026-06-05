# ADR-017 — Trust Store Lives in User App-Data, Not the Install Directory

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

NNSpire maintains a live PKI trust store (ADR-007) that must be:
- **Writable at runtime** — Trust Update Packages (TUPs) add/update certificates and CRL caches.
- **User-scoped** — different OS users on the same machine should have independent trust
  stores (enterprise policy may differ per user).
- **Persistent across Studio updates** — a Studio reinstall must not wipe the user's
  accumulated trust state (applied TUPs, enterprise CA imports).
- **Protected against concurrent writes** — multiple Studio instances (or an update process)
  must not corrupt the store.

The install directory (`Program Files\`, `/usr/share/`, `/Applications/`) is typically:
- Read-only for non-admin users on all major platforms.
- Replaced or wiped on Studio update/reinstall.

Neither property is acceptable for a live, writable trust store.

---

## Decision

The trust store lives in the **platform user application-data directory**, never in the
install directory.

```
Windows:   %APPDATA%\NNSpire\truststore\
macOS:     ~/Library/Application Support/NNSpire/truststore/
Linux:     ~/.local/share/NNSpire/truststore/
```

Structure:
```
truststore/
├── roots/              ← PEM files; seeded on first run from binary-embedded seed_root_ca.pem
├── intermediates/      ← Plugin Registry CA and any imported intermediates
├── enterprise/         ← Enterprise Intermediate CAs (imported via TUP)
├── crls/               ← Cached CRLs, refreshed on network access
├── ocsp_cache/         ← OCSP response staples
├── history/            ← Append-only JSON log of all trust operations
└── store.lock          ← Exclusive file lock held during any write
```

### First-run seeding

On first run, if `roots/` is empty, the Studio copies the `seed_root_ca.pem`
embedded in the binary into `roots/root_ca_v1.pem`. This is the only time the
install directory influences the trust store.

### Permissions

- `roots/`, `intermediates/`, `enterprise/`: owner read/write only (`chmod 600` / Windows ACL equivalent).
- `history/`: append-only; the application never deletes or modifies existing entries.
- `store.lock`: exclusive file lock for any write operation; prevents corruption from concurrent instances.

### Portable / enterprise override

An enterprise administrator may override the trust store path via a registry key (Windows)
or environment variable (`NNSpire_TRUSTSTORE_PATH`) to redirect to a shared network
location for centralised management.

---

## Consequences

**Positive**
- Trust store survives Studio reinstalls and updates.
- Each OS user has an independent trust state.
- Append-only history provides a full audit trail of trust operations.
- The portable path override satisfies enterprise deployment scenarios.

**Negative / constraints**
- Trust store is not backed up by default — users should be advised to include
  `%APPDATA%\NNSpire\` in their backup strategy.
- `store.lock` must be correctly handled to avoid deadlock if Studio crashes mid-write.
- The enterprise path override adds a security consideration: the redirected path must
  itself be protected against tampering.

**Follow-on**
- `TrustStore` class in `NNSpire/plugin-api/trust/`.
- Path resolution in `TrustStore::defaultPath()` using `QStandardPaths::AppDataLocation`.
- Lock implementation: `QLockFile` (Qt) or platform `flock`/`LockFileEx`.
- See TRUST-ARCHITECTURE.md §3 for the full trust store specification.
