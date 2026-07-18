# ADR-052 — NNSpire Agent Structural Architecture (Decisions 4-15)

**Date:** 2026-07-16
**Status:** Accepted
**Decided by:** Project founder (architectural discussion session)
**Related:** [`ADR-050`](ADR-050-nnagent-ui-framework-tauri.md), [`ADR-051`](ADR-051-nnagent-implementation-language-scoped.md)

---

## Consolidated Decisions #4 through #15

### D4: Storage Backend — JSON-First, SQLite for Multi-User Only

**Primary storage:** Flat JSON files for single-user mode. Zero database dependency.

```
~/.nnspire/nnagent/
├── settings.json              # Main app settings
├── providers.json             # Model provider configs
├── profiles.json              # User profiles
├── templates.json             # Prompt templates
├── mcp.json                   # MCP server configs
├── kb.json                    # Knowledge base configs
├── automations/               # One JSON per automation template
│   └── auto_report.json
├── conversations/             # Mirrors user folder structure
│   ├── .config                # Folder metadata (sync source, auth info)
│   ├── .cache/                # Cached data if not authoritative source
│   ├── chat_001.json          # Single conversation
│   └── Work/                  # Subfolder
│       ├── .config
│       ├── chat_002.json
│       └── Projects/
│           ├── .config
│           └── chat_003.json
├── folders.json               # Folder structure index
└── users.json                 # Multi-user mode only
```

**Folder `.config` format:**
```json
{
  "folderId": "work-projects",
  "displayName": "Projects",
  "authoritativeSource": "local",
  "syncConfig": null,
  "cachedFrom": null,
  "lastSynced": null,
  "parentFolderId": "work"
}
```

**SQLite role:** ONLY in multi-user mode for:
- User ↔ Group relations
- Group ↔ Authorization mappings
- Folder ACLs
- Concurrent access locking

**DB portability:** JSON schema designed with relational mapping in mind. Each JSON file = one table. Attributes = columns. Nested arrays = child tables with foreign keys.

---

### D5: Plugin SDK — Soft PKI with Sandboxing

| Plugin Signature | Trust Level | Permissions |
|-----------------|-------------|-------------|
| NNSpire-signed | Full trust | All APIs |
| Self-signed | Warning on load | All APIs |
| Unsigned | Sandboxed | Restricted FS/network |

Enterprise deployments can toggle "strict mode" (NNSpire-signed only).

**Same PKI infrastructure as ADR-007**, but permissive by default.

---

### D6: MCP Architecture — Dual Client+Server, AIAPI Source Bundled

```
┌─────────────────────────────────────────────────┐
│              NNSpire Agent                        │
│                                                   │
│  ┌─────────────┐    ┌─────────────────────────┐ │
│  │  MCP Client  │───▶│  Tool Registry           │ │
│  │  (outgoing)  │    │  - Web Fetch (bundled)   │ │
│  └─────────────┘    │  - File Read/Write        │ │
│                     │  - Terminal Exec           │ │
│  ┌─────────────┐    │  - UI Automation (bundled) │ │
│  │  MCP Server  │◀───│  - Custom (plugins)      │ │
│  │  (incoming)  │    └─────────────────────────┘ │
│  │  for Studio  │                                │
│  └─────────────┘                                │
└─────────────────────────────────────────────────┘
```

**AIAPI MCP source code bundled** within NNAgent distribution:
- Web Fetch helper → compiled into NNAgent
- UI Automation helper → KeyWin.exe bundled
- Browser helper → BrowserWin.exe bundled
- Security filter chain → compiled into NNAgent

**External AIAPI server override:** If `mcp.json` specifies an external AIAPI URL, bundled tools are bypassed in favor of remote calls. Configuration flags in both AIAPI and NNAgent allow coordinated operation.

**Transport:** HTTP/SSE for remote, stdio for local.
**Protocol:** MCP 2024-11-05 (current stable).

---

### D7: Graph Editor — React Flow (xyflow)

- **Library:** [@xyflow/react](https://www.npmjs.com/package/@xyflow/react)
- **Home:** [reactflow.dev](https://reactflow.dev)
- **Rationale:** 14K+ GitHub stars, TypeScript-native, touch-friendly, custom HTML nodes, used by Mermaid/LangFlow/FlowiseAI

---

### D8: Container UI — React Web UI

Same React codebase as desktop (`desktop/src/` shared with `web/src/`). Served via REST API server in Docker container.

---

### D9: Library Split — Core vs App

```
nnagent/
├── core/              # nnagent-core.lib/.a — C++ static library
├── desktop/           # Tauri app (Rust FFI → core)
├── web/               # React web UI (REST API → core)
├── cli/               # Headless CLI (static link to core)
└── studio_embed/      # QWebEngineView wrapper
```

---

### D10: Engine Communication — HTTP/REST + IPC

- **Primary:** HTTP/REST via Runner API
- **Local optimization:** Named pipes (Windows) / Unix sockets (Linux/macOS)
- **Protocol:** JSON-RPC 2.0

---

### D11: Vector DB — Qdrant Default Plugin

```
IVectorDB (interface)
├── QdrantPlugin (default — ships with distro)
├── ChromaPlugin (future)
└── WeaviatePlugin (future)
```

---

### D12: Auth System — Mode-Dependent

| Mode | Auth | Users | RBAC |
|------|------|-------|------|
| Single-user | None | Implicit | N/A |
| Local service | OS auth | Local | Basic |
| Remote server | OAuth/LDAP | Full | Full |

Password hashing: Argon2id. Initial OAuth: Google, Microsoft, GitHub.

---

### D13: Theming — JSON Theme Descriptor ⚠️ NEW

Shared theme format consumed by both Studio (QML) and Agent (React):

```json
{
  "name": "nnspire-dark",
  "version": "1.0",
  "tokens": {
    "color-primary": "#0078d4",
    "color-background": "#1e1e1e",
    "color-surface": "#2d2d2d",
    "font-family": "Segoe UI, system-ui",
    "font-size-base": 14,
    "border-radius": 6
  },
  "skin": {
    "showSidebar": true,
    "showStatusBar": true,
    "compactMode": false
  }
}
```

- Studio: JSON → QML `ThemePlugin`
- Agent: JSON → CSS `:root` custom properties
- Same token vocabulary, different renderers

---

### D14: Script Engine — Node.js Subprocess ⚠️ NEW

| Language | Engine | Integration |
|----------|--------|-------------|
| JavaScript/TypeScript | Node.js subprocess | Full npm support |
| Python | Python subprocess | Automation scripts |
| C++ Plugin | Dynamic library load | ADR-003 |

Rationale: Full npm ecosystem, process isolation, cross-platform consistency.

---

### D15: Deployment Order

| Phase | Target | Deliverable |
|-------|--------|-------------|
| 1 | Desktop | Tauri app (Win/Mac/Linux) |
| 2 | CLI | Headless C++ binary |
| 3 | Service/Daemon | Windows Service + systemd |
| 4 | Container | Docker + React web UI |
| 5 | Mobile | Capacitor (Android/iOS) |
| 6 | Office Plugin | COM/UNO (future) |

Office plugin: Out of scope for initial release.

---

## Consequences

### Positive
- JSON-first storage simplifies backup/sync/user management
- Bundled AIAPI source makes NNAgent self-contained
- React Flow provides best-in-class graph editing
- JSON theme descriptor enables cross-pillar theming
- Node.js subprocess gives full npm automation support

### Negative / Constraints
- JSON file I/O slower than DB for large conversation histories
- Bundled AIAPI increases distribution size (~5-10 MB)
- React Flow requires npm dependency management
- Node.js subprocess adds ~50 MB RAM per automation run

### Follow-on
- Storage layer must implement JSON read/write with file locking
- AIAPI source integration requires license compatibility check
- React Flow custom nodes for automation blocks need design spec
- Theme descriptor schema needs formal JSON Schema definition
- Node.js subprocess lifecycle management needs design
