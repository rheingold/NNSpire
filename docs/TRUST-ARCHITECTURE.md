# NNSpire — Trust Architecture Whitepaper

**Version**: 0.1 (Phase 0 — design)  
**Date**: 2026-03-31  
**Status**: DRAFT — pre-implementation baseline

---

## 1. Motivation

NNSpire loads third-party plugins at runtime. Plugins execute arbitrary code inside the Studio process (or optionally in the `NNSpire-runner` sidecar). Without a trust framework, any malicious plugin could compromise the user's system, exfiltrate model weights, or corrupt training data.

The trust framework solves three problems simultaneously:
1. **Security**: prevent unsigned or compromised plugins from loading silently.
2. **Economy**: provide a registration mechanism that gives plugin authors an economic path (commercial signing) while keeping the mechanism itself open source.
3. **Continuity**: ensure no installed Studio version ever becomes permanently stranded by a key compromise or key rotation — trust anchors are updatable from day zero.

---

## 2. Certificate hierarchy

```
Root CA  (project owner — kept offline, HSM recommended)
│  Self-signed; validity 20 years
│  Published openly as seed_root_ca.pem
│  Embedded in Studio binary as trust seed
│
├── NNSpire Plugin Registry CA  (Intermediate)
│   │  Signed by Root CA; validity 5 years
│   │  Used to issue Vendor and Community signing certs
│   │
│   ├── Community Signing Cert  (issued per open-source plugin author)
│   │   Validity: 2 years, auto-renewable
│   │   NameConstraints: O = author's registered handle
│   │   EKU: NNSpirePluginSigning (OID 1.3.6.1.4.1.NNSpire.1.1)
│   │
│   ├── Vendor Signing Cert  (issued per commercial plugin author)
│   │   Validity: 1–3 years, manually renewed
│   │   NameConstraints: O = company name
│   │   EKU: NNSpirePluginSigning
│   │
│   └── Trust Update Package Signing Cert
│       Validity: 3 years
│       EKU: NNSpireTrustUpdate (OID 1.3.6.1.4.1.NNSpire.1.2)
│
└── Enterprise Intermediate CA  (issued per qualifying organisation)
    │  Signed by Root CA; validity 1–3 years
    │  PathLenConstraint: 0 (no further sub-CAs)
    │  NameConstraints: O = <company name exactly>
    │  EKU: NNSpirePluginSigning, NNSpireEnterpriseCA (OID 1.3.6.1.4.1.NNSpire.1.3)
    │
    └── Enterprise Plugin Signing Cert  (self-issued by the enterprise)
        Validity: enterprise's choice
        Trusted on machines that have imported the Enterprise Intermediate CA via TUP
```

### Custom OID registry

| OID | Name | Purpose |
|---|---|---|
| `1.3.6.1.4.1.NNSpire.1.1` | `NNSpirePluginSigning` | EKU: authorises signing plugin binaries |
| `1.3.6.1.4.1.NNSpire.1.2` | `NNSpireTrustUpdate` | EKU: authorises signing Trust Update Packages |
| `1.3.6.1.4.1.NNSpire.1.3` | `NNSpireEnterpriseCA` | EKU: marks an Enterprise Intermediate CA cert |

> NNSpire placeholder IANA PEN — obtain a real Private Enterprise Number at https://www.iana.org/assignments/enterprise-numbers before first release.

---

## 3. Trust Store

The live trust store is maintained in the user's application data directory, never inside the installation folder (which may be read-only).

```
<app_data>/NNSpire/truststore/
├── roots/
│   └── root_ca_v1.pem          ← seeded on first run from embedded seed_root_ca.pem
├── intermediates/
│   └── plugin_registry_ca.pem
├── enterprise/
│   └── acme_corp_intermediate.pem   ← after enterprise TUP acceptance
├── crls/
│   └── plugin_registry_ca.crl   ← cached CRL, refreshed on network access
├── ocsp_cache/
│   └── <cert_serial>.ocsp       ← OCSP response staples
├── history/
│   └── 2026-03-31T00:00:00Z_seed.log
│   └── 2026-04-01T12:00:00Z_tup_applied.log
└── store.lock                   ← prevents concurrent writes
```

### Properties

- `roots/`, `intermediates/`, `enterprise/`: PEM files only. Permissions: owner read/write on all platforms (`chmod 600` / Windows ACL equivalent).
- `history/`: **append-only**. The application never deletes or modifies entries. Each line is a JSON record: `{ "timestamp", "action": "SEED|TUP_APPLY|CERT_ADD|CERT_REVOKE", "detail", "operator" }`.
- `store.lock`: exclusive file lock held during any write operation. Prevents corruption from concurrent Studio instances.

### First-run seeding

```cpp
if (!trustStoreExists()) {
    TrustStore::seed(QCoreApplication::applicationDirPath()
                     + "/seed_roots/root_ca_v1.pem");
    // writes to <app_data>/truststore/roots/root_ca_v1.pem
    // appends seed event to history/
}
```

---

## 4. Trust verification flow (plugin load)

```
PluginLoader::load(manifestPath)
    │
    ├─ 1. Parse plugin.manifest.json (no code loaded)
    │
    ├─ 2. Read detached signature .p7s
    │
    ├─ 3. TrustVerifier::verify(manifest_bytes, signature, trustStore)
    │       ├─ OpenSSL CMS_verify: validate PKCS#7 against chain in trust store
    │       ├─ Check EKU = NNSpirePluginSigning
    │       ├─ CRL check: look up issuer CRL in crls/ (download if stale, skip if offline)
    │       ├─ OCSP check (if not offline): consult issuer OCSP responder; cache result
    │       ├─ Timestamp check: signing time ≤ now ≤ cert expiry (+ 3-day clock skew tolerance)
    │       └─ NameConstraints: signer O= matches what issuer declared
    │
    ├─ 4. Determine trust level:
    │       TRUSTED     → cert chain valid, not revoked
    │       COMMUNITY   → cert chain valid, community cert type
    │       UNVERIFIED  → no cert, self-signed, or unknown root
    │       REVOKED     → cert explicitly revoked in CRL/OCSP
    │
    ├─ 5. Act on trust level:
    │       TRUSTED / COMMUNITY → proceed to load
    │       UNVERIFIED → emit signal to UI; block until user clicks "Load anyway";
    │                    re-prompt every session start; show persistent yellow banner
    │       REVOKED    → emit signal to UI; hard block; no force-load path
    │
    └─ 6. Load code (QPluginLoader or Python importer)
```

The `TrustUpdateHandler` is NOT part of `PluginLoader`. It is a separate class compiled directly into the core binary. This prevents any plugin from intercepting or hijacking the trust update mechanism.

---

## 5. Trust Update Package (TUP)

### File format

A TUP is a signed zip archive with the extension `.NNSpire-trust`:

```
trust-update-package/
├── manifest.json         ← TUP metadata
├── certs/
│   ├── add/              ← PEM certs to add (roots or intermediates)
│   └── revoke.json       ← list of { issuerDN, serialNumber } to revoke locally
└── signature.p7s         ← PKCS#7 detached signature over canonical JSON of manifest
                             + SHA256 hashes of all certs in certs/add/
```

### TUP `manifest.json`

```json
{
  "tupVersion": 1,
  "type": "TRUST_UPDATE",
  "id": "uuid-v4",
  "timestamp": "2026-03-31T00:00:00Z",        ← must be > last applied TUP timestamp
  "issuer": "CN=NNSpire Plugin Registry CA",
  "purpose": "Add Enterprise CA for ACME Corp",
  "certsToAdd": [
    { "file": "certs/add/acme_corp_intermediate.pem", "sha256": "<hex>", "type": "ENTERPRISE_CA" }
  ],
  "certsToRevoke": [],
  "minStudioVersion": "1.0.0"
}
```

### Validation rules (ALL must pass before applying)

| # | Rule | Rationale |
|---|---|---|
| 1 | Signature valid against current trust store | TUP must be signed by a currently trusted cert with EKU `NNSpireTrustUpdate` |
| 2 | `timestamp` > last applied TUP timestamp | Prevents replay/downgrade attacks |
| 3 | No cert being added is the signing cert itself | Prevents self-escalation |
| 4 | A TUP cannot revoke the cert it was signed with | Prevents lockout |
| 5 | `minStudioVersion` ≤ current Studio version | Prevents applying future TUPs to old binaries |
| 6 | **User confirmation modal** required — not dismissable programmatically | User must explicitly approve every trust store mutation |

### Application (atomic)

```
1. Write new cert(s) to <truststore>/staging/
2. Verify all validation rules
3. Show user confirmation modal with full diff (added certs, revoked serials, purpose)
4. On user confirm:
   a. Append history log entry BEFORE any write (crash-safe marker)
   b. Rename staging/ files into roots/ or intermediates/ or enterprise/
   c. Append revoked serials to local revocation list
   d. Append completion log entry
5. On reject: delete staging/, append "TUP_REJECTED" history entry
```

---

## 6. Enterprise Intermediate CA workflow

**For an organisation that wants to self-sign plugins internally:**

1. Organisation generates a key pair and CSR:
   ```bash
   NNSpire-sign keygen --type enterprise-ca --org "ACME Corp" --out acme-ca/
   ```

2. Organisation submits CSR to NNSpire Plugin Registry via `NNSpire-sign issue-enterprise-ca` or the registry web portal (admin-reviewed, commercial process):
   ```bash
   NNSpire-sign issue-enterprise-ca --csr acme-ca/enterprise.csr \
     --registry https://registry.NNSpire.dev --token <admin-token>
   ```

3. Project owner reviews, approves, and the registry returns a TUP containing the issued Enterprise Intermediate CA cert.

4. Organisation distributes this TUP to all their employees' Studio installs (email, IT MDM, intranet portal). Studio shows confirmation modal; employees accept.

5. Organisation self-signs their internal plugins:
   ```bash
   NNSpire-sign sign --manifest plugin.manifest.json \
     --cert acme-intermediate.crt --key acme-intermediate.key \
     --out signed/
   ```

6. Internal employees get `TRUSTED` load (silent) for all ACME-signed plugins. External users get `UNVERIFIED` (they have not accepted the ACME TUP).

---

## 7. Root CA rotation

If the Root CA key is compromised or reaches end-of-life:

1. Generate new Root CA key pair (offline, airgapped recommended).
2. Old Root CA signs a TUP adding the new Root CA cert to `roots/`.
3. Distribute TUP to all users. Users accept via confirmation modal.
4. After a declared transition window, old Root CA signs a second TUP revoking itself from `roots/`.
5. All new intermediate certs and plugin signing certs are issued by the new Root CA.
6. Old plugins signed under the old chain continue to validate as long as:
   - The plugin's signing cert was valid at signing time (timestamp-pinned),
   - The old Root CA cert is still in the local trust store (it stays there until the self-revocation TUP is accepted by the user — the user controls the timeline with full information).

---

## 8. Security hardening checklist

- [ ] `TrustUpdateHandler` compiled directly into Studio binary (not loadable as plugin)
- [ ] Trust store directory: owner-only permissions on all platforms
- [ ] `history/` is append-only; no delete or modify code path exists in the binary
- [ ] TUP application is atomic (temp → verify → rename; crash leaves staging/ which is cleaned on next start)
- [ ] No eval / dynamic code in TrustVerifier; OpenSSL used as the validation library
- [ ] Root CA private key never touches a networked machine (documented requirement)
- [ ] CRL refresh is silent and non-blocking; offline operation never blocked
- [ ] OCSP responses cached with `nextUpdate` expiry; stale cache used if network unavailable
- [ ] `NNSpire-runner` sidecar (when enabled) runs with reduced OS privileges (AppContainer on Windows, seccomp on Linux, App Sandbox on macOS)
- [ ] SBOM generated for each release including OpenSSL version
- [ ] Reproducible builds goal: same source + toolchain → byte-identical binary (important for auditing embedded `seed_roots/root_ca.pem` and trust seed logic)
