# NNSpire Agent (`nnagent`) — Architectural Recommendations & Clarifications

> **Created:** 2026-07-16
> **Purpose:** Resolve ambiguities before implementation begins. Each item below represents a decision point that MUST be addressed before coding starts.

---

## 1. PLATFORM & FRAMEWORK DECISIONS (CRITICAL)

### 1.1 Primary UI Framework — Qt6/QML vs Electron vs Tauri

**Current State:** The requirements document (section 3) asks "Qt, modern Electron+Rust or similar smaller node.js app?" — this is UNRESOLVED.

**Existing Project Context:**
- [`ADR-001`](docs/adr/ADR-001-qt6-ui-framework.md): Qt 6 is THE GUI framework for NNSpire
- [`ADR-018`](docs/adr/ADR-018-qml-not-qt-widgets.md): QML (Qt Quick) mandated, not Qt Widgets
- [`ADR-002`](docs/adr/ADR-002-cpp17-engine-language.md): C++17 is the core language
- [`ADR-004`](docs/adr/ADR-004-dual-language-everywhere.md): Dual-language parity (C++ + Python)

**Recommendation:** **Qt6/QML for NNAgent** — for the following reasons:

| Criterion | Qt6/QML | Electron | Tauri |
|-----------|---------|----------|-------|
| Consistency with NNSpire | ✅ Same framework | ❌ Different | ❌ Different |
| Shared code with Studio | ✅ Reusable QML components | ❌ No | ❌ No |
| Embedded Agent Panel in Studio | ✅ Native | ❌ iframe hack | ❌ iframe hack |
| Binary size | ~30-50 MB | ~150-200 MB | ~5-15 MB |
| Cross-platform | ✅ Win/macOS/Linux/iOS/Android | ✅ Win/macOS/Linux | ✅ Win/macOS/Linux |
| Mobile support | ✅ Native | ❌ No | ⚠️ Limited |
| C++ integration | ✅ Direct | ❌ IPC only | ✅ Rust FFI |
| Learning curve | Medium (already in project) | Low | Medium |

**Decision Required:**
- [ ] **CONFIRM: Qt6/QML for NNAgent** OR provide justification for alternative
- [ ] If alternative chosen: HOW will "Embedded Agent Panel in Studio" work?

### 1.2 Implementation Language

**Recommendation:** **C++17 primary, Python mirror** — following ADR-004 dual-language mandate.

| Layer | Language | Rationale |
|-------|----------|-----------|
| Core logic | C++17 | Consistency with engine |
| UI | QML + C++ controllers | ADR-018 |
| Python bindings | pybind11 | ADR-005 |
| Automation scripts | JavaScript/Python | User-defined |
| Plugin interface | C ABI + Python | ADR-003 |

**Decision Required:**
- [ ] **CONFIRM: C++17 primary** OR specify alternative
- [ ] Should NNAgent follow ADR-004 dual-language parity? (Y/N)

---

## 2. PROJECT STRUCTURE

### 2.1 Where does `nnagent/` live?

**Current State:** `.gitignore` excludes `nnagent/` entirely. The folder existed but was not tracked.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| A. Inside `nnspire/` mono-repo | Single build, shared deps | Blurs pillar boundaries |
| B. `nnagent/` at repo root | Clear separation | Separate build config |
| C. Separate repository | Full independence | Harder coordination |

**Recommendation:** **Option B — `nnagent/` at repo root** — maintains pillar separation while allowing shared documentation and coordinated releases.

**Decision Required:**
- [ ] **CONFIRM: `nnagent/` at repo root** (and remove from `.gitignore`)
- [ ] Should `nnagent/` have its own `CMakeLists.txt` or be part of root CMake?

### 2.2 Build System Integration

**Existing Context:** [`ADR-010`](docs/adr/ADR-010-cmake-build-system.md) mandates CMake 3.21+

**Recommendation:** NNAgent should be a CMake sub-project that can be built independently OR as part of the full NNSpire build.

```
NNSpire/
├── CMakeLists.txt          # Root — conditionally includes nnagent/
├── nnspire/
│   └── CMakeLists.txt      # Engine + Studio
└── nnagent/
    └── CMakeLists.txt      # Agent (optional build)
```

**Decision Required:**
- [ ] **CONFIRM: CMake for NNAgent**
- [ ] `BUILD_NNAGENT` CMake option to toggle agent build?

---

## 3. DATA STORAGE

### 3.1 Primary Storage Backend

**Requirements Analysis:**
- Single-user mode: local files
- Multi-user mode: database
- Remote sync: NFS/FTP/WebDAV/OneDrive/Google Drive
- Chat history, profiles, automations, KB, MCP configs

**Recommendation:** **SQLite for local, PostgreSQL for server** — hybrid approach.

| Mode | Storage | Rationale |
|------|---------|-----------|
| Single-user desktop | SQLite | Zero config, portable |
| Local file fallback | JSON/XML files | Export/import, human readable |
| Multi-user server | PostgreSQL | ACID, concurrent access |
| Remote sync | Abstracted provider | Plugin-based |

**Decision Required:**
- [ ] **CONFIRM: SQLite primary for local**
- [ ] Should JSON/XML be the canonical format with SQLite as cache?
- [ ] PostgreSQL for multi-user mode — required now or future?

### 3.2 Configuration File Format

**Recommendation:** **JSON for configuration, SQLite for runtime data**

| What | Format | Location |
|------|--------|----------|
| Main settings | JSON | `~/.nnspire/nnagent/settings.json` |
| Model providers | JSON | `~/.nnspire/nnagent/providers.json` |
| Profiles | JSON | `~/.nnspire/nnagent/profiles.json` |
| Chat history | SQLite | `~/.nnspire/nnagent/history.db` |
| Automations | JSON per file | `~/.nnspire/nnagent/automations/` |
| KB config | JSON | `~/.nnspire/nnagent/kb.json` |
| MCP config | JSON | `~/.nnspire/nnagent/mcp.json` |
| Prompt templates | JSON | `~/.nnspire/nnagent/templates.json` |
| User/auth | SQLite/JSON | Mode-dependent |

**Decision Required:**
- [ ] **CONFIRM: JSON for config files**
- [ ] Storage path — `~/.nnspire/nnagent/` or elsewhere?

---

## 4. PLUGIN ARCHITECTURE

### 4.1 Plugin Types for NNAgent

**Required Plugin Interfaces:**

| Plugin Type | Purpose | Interface |
|-------------|---------|-----------|
| Model Provider | LLM API calls | `IModelProvider` |
| File Parser | Binary→Text conversion | `IFileParser` |
| Vector DB | RAG storage | `IVectorDB` |
| KB Source | Data source connector | `IKBSource` |
| Automation Block | Custom graph block | `IAutomationBlock` |
| Storage Backend | Chat/config storage | `IStorageBackend` |
| Sync Provider | Remote sync | `ISyncProvider` |

**Decision Required:**
- [ ] **CONFIRM: Plugin types list** (add/remove)
- [ ] Should NNAgent plugins reuse NNSpire plugin SDK (signing, trust)?
- [ ] Plugin loading: dynamic library (.dll/.so) or Python (.pyd)?

### 4.2 Plugin Security

**Recommendation:** Reuse NNSpire PKI trust chain from [`ADR-007`](docs/adr/ADR-007-pki-trust-chain.md).

- Plugins signed with same certificate hierarchy
- `TrustStore` validates before loading
- Sandboxed execution for untrusted plugins

**Decision Required:**
- [ ] **CONFIRM: Reuse NNSpire PKI for NNAgent plugins**

---

## 5. MCP INTEGRATION

### 5.1 MCP Client Architecture

**Context:** NNAgent MUST be an MCP client (to call tools) AND can expose MCP server interface (for Studio integration).

**Recommendation:**

```
┌─────────────────────────────────────────────────┐
│              NNAgent Application                 │
│                                                  │
│  ┌─────────────┐    ┌─────────────────────────┐ │
│  │  MCP Client  │───▶│  Tool Registry          │ │
│  │  (outgoing)  │    │  - Web Fetch             │ │
│  └─────────────┘    │  - File Read/Write       │ │
│                     │  - Terminal Exec          │ │
│  ┌─────────────┐    │  - Custom (plugins)      │ │
│  │  MCP Server  │◀───│                         │ │
│  │  (incoming)  │    └─────────────────────────┘ │
│  │  for Studio  │                                │
│  └─────────────┘                                │
└─────────────────────────────────────────────────┘
```

**Decision Required:**
- [ ] **CONFIRM: Dual MCP client+server architecture**
- [ ] MCP protocol version to target?
- [ ] Transport: stdio, HTTP/SSE, or both?

---

## 6. AUTOMATION ENGINE

### 6.1 Graph Editor Technology

**Requirement:** Visual graph editor with blocks, connections, parameters.

**Options:**

| Technology | Pros | Cons |
|------------|------|------|
| QML + Canvas | Native, GPU accelerated | Custom implementation |
| QML Graph Editor libraries | Reusable | Limited options |
| Web-based (if Electron) | Many libraries | Not Qt |
| Custom C++ | Full control | Most work |

**Recommendation:** **QML Canvas with custom graph editor** — consistent with ADR-018.

**Decision Required:**
- [ ] **CONFIRM: QML Canvas for graph editor**
- [ ] Any existing QML graph editor library to evaluate?

### 6.2 Script Execution in Automation Blocks

**Requirement:** Script blocks supporting JavaScript/TypeScript/Python.

**Recommendation:**

| Language | Engine | Integration |
|----------|--------|-------------|
| JavaScript | Qt Quick JS (QJSEngine) | Native in Qt |
| Python | Python subprocess or QPython | ADR-004 parity |
| C++ Plugin | Dynamic library load | ADR-003 |

**Decision Required:**
- [ ] **CONFIRM: QJSEngine for JavaScript**
- [ ] Python via subprocess or embedded?
- [ ] Node.js/npm package support for automation scripts?

---

## 7. DEPLOYMENT MODELS

### 7.1 Deployment Matrix

**Required:**

| Mode | Platform | Implementation |
|------|----------|----------------|
| Desktop App | Windows | Qt6 executable + installer |
| Desktop App | macOS | Qt6 app bundle + DMG |
| Desktop App | Linux | Qt6 executable + AppImage/Deb |
| CLI | All | Headless Qt or separate binary |
| Container | Docker | Qt WebAssembly or separate web UI |
| Service | Windows | Windows Service |
| Service | Linux/macOS | systemd/launchd |
| Mobile | iOS | Qt6 iOS deployment |
| Mobile | Android | Qt6 Android deployment |
| Office Plugin | Win/Mac/Linux | COM/MacScript/UNO |

**Recommendation:** Phased approach — Desktop first, then Service, then Container, then Mobile.

**Decision Required:**
- [ ] **CONFIRM: Phased deployment order**
- [ ] Container deployment — Qt WebAssembly or React web UI?
- [ ] Office plugin — in scope for initial release?

### 7.2 Container Deployment Ambiguity

**Problem:** Qt is NOT suitable for web containers. The requirements mention "container with web UI."

**Options:**

| Option | Description | Trade-off |
|--------|-------------|-----------|
| A. Separate web UI | React/Vue frontend + REST API | Duplication of UI logic |
| B. Qt WebAssembly | Compile Qt to WASM | Large size, limited features |
| VNC/Remote Desktop | Stream Qt UI | Poor UX |

**Recommendation:** **Option A — Separate web UI sharing backend API** — the REST API becomes the single source of truth.

**Decision Required:**
- [ ] **CONFIRM: Separate web UI for container** OR specify alternative
- [ ] Web UI framework preference (React, Vue, Svelte)?

---

## 8. CROSS-PILLAR INTEGRATION

### 8.1 Embedded Agent Panel in Studio

**Requirement:** NNAgent must be embeddable in NNSpire Studio.

**Implication:** NNAgent core logic MUST be extractable as a shared library.

**Recommended Architecture:**

```
nnagent/
├── core/             # nnagent-core.lib/.so — reusable library
│   ├── llm_router/
│   ├── mcp_client/
│   ├── chat_engine/
│   └── automation/
├── app/              # Standalone NNAgent desktop app
│   └── main.cpp
├── web/              # Web UI (for container deployment)
│   └── ...
└── studio_plugin/    # Qt plugin for embedding in Studio
    └── AgentPanel.qml
```

**Decision Required:**
- [ ] **CONFIRM: Library split pattern** (core vs app)
- [ ] How does Studio load the Agent Panel? (Qt plugin, direct link?)

### 8.2 Communication with NNSpire Engine

**Existing:** ADR mandates Agent reaches Engine through Runner API only.

**Recommendation:** HTTP/REST primary, IPC (named pipes/shared memory) for local performance.

**Decision Required:**
- [ ] **CONFIRM: HTTP/REST + IPC for Engine communication**
- [ ] Runner API specification location?

---

## 9. KNOWLEDGE BASE & RAG

### 9.1 Vector Database Plugin Architecture

**Recommendation:**

```
IKBSource (interface)
├── FileSystemSource (default)
├── DatabaseSource (future)
├── RegistrySource (future)
└── PIMSource (future)

IVectorDB (interface)
├── QdrantPlugin (default)
├── ChromaPlugin (future)
└── WeaviatePlugin (future)

IFileParser (interface)
├── TextParser (default)
├── PDFParser
├── DOCXParser
├── ImageParser (OCR)
└── AudioParser (transcription)
```

**Decision Required:**
- [ ] **CONFIRM: Plugin architecture for KB/RAG**
- [ ] Qdrant as default — required now or future?
- [ ] Embedding model — local or API-based?

---

## 10. MULTI-USER & AUTHORIZATION

### 10.1 Auth Architecture

**Requirement:** SSO, users, groups, authorizations.

**Recommendation:**

| Mode | Auth | Users |
|------|------|-------|
| Single-user | None (auto-login) | Implicit single user |
| Local service | OS auth | Local users |
| Remote server | OAuth/LDAP/Windows | Full RBAC |

**Decision Required:**
- [ ] **CONFIRM: Mode-dependent auth**
- [ ] OAuth providers to support initially?
- [ ] LDAP/Active Directory — required now or future?

---

## 11. THEMING & SKINNING

### 11.1 Theme System

**Context:** Requirements mention "ThemeDescriptor" and "skin descriptor."

**Recommendation:** Reuse NNSpire Studio theming system.

| Concept | Implementation |
|---------|----------------|
| Theme | Color palette, fonts, spacing |
| Skin | Layout customization, component visibility |
| Hot-reload | QML property binding + file watcher |

**Decision Required:**
- [ ] **CONFIRM: Shared theming with Studio**
- [ ] Theme file format — JSON, QML, or CSS-like?

---

## 12. RESOLVING AMBIGUITIES — SUMMARY TABLE

| # | Ambiguity | Recommendation | Status |
|---|-----------|----------------|--------|
| 1 | UI Framework | Qt6/QML | ⚠️ CONFIRM |
| 2 | Primary Language | C++17 | ⚠️ CONFIRM |
| 3 | Project Location | `nnagent/` at root | ⚠️ CONFIRM |
| 4 | Storage Backend | SQLite + JSON | ⚠️ CONFIRM |
| 5 | Plugin SDK | Reuse NNSpire PKI | ⚠️ CONFIRM |
| 6 | MCP Architecture | Client + Server | ⚠️ CONFIRM |
| 7 | Graph Editor | QML Canvas | ⚠️ CONFIRM |
| 8 | Container UI | Separate web UI | ⚠️ CONFIRM |
| 9 | Library Split | Core vs App | ⚠️ CONFIRM |
| 10 | Engine Comms | HTTP + IPC | ⚠️ CONFIRM |
| 11 | Vector DB | Qdrant default | ⚠️ CONFIRM |
| 12 | Auth System | Mode-dependent | ⚠️ CONFIRM |
| 13 | Theming | Shared with Studio | ⚠️ CONFIRM |
| 14 | Script Engine | QJSEngine | ⚠️ CONFIRM |
| 15 | Deployment Order | Desktop→Service→Container→Mobile | ⚠️ CONFIRM |

---

## 13. WHAT TO LEAVE FOR LARGER MODEL (Claude/GPT-4)

If the above decisions are confirmed, the following items may benefit from deeper analysis by a larger model:

1. **Detailed API specification** for each plugin interface
2. **Database schema design** for SQLite/PostgreSQL
3. **Security threat model** for multi-user deployments
4. **Performance requirements** for automation engine
5. **Mobile UI/UX design** specifics
6. **Office plugin architecture** details
7. **CI/CD pipeline** configuration
8. **Testing strategy** at integration level

---

## 14. NEXT STEPS

1. **Architect reviews** this document and confirms/modifies recommendations
2. **ADRs created** for NNAgent-specific decisions (ADR-050+)
3. **`nnagent-TODO.md`** updated with confirmed decisions
4. **Implementation begins** with Phase 1 (Foundation)

---

*This document is a LIVING document — update as decisions are made.*
