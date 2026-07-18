# NNSpire Agent (`nnagent`)

**Conversational AI Workbench** — Third pillar of the NNSpire project.

> **Pillar**: NNSpire Agent (L6 application task flow)  
> **Framework**: Tauri 2.x + React/TypeScript + C++17 Core  
> **ADRs**: [ADR-050](../docs/adr/ADR-050-nnagent-ui-framework-tauri.md), [ADR-051](../docs/adr/ADR-051-nnagent-implementation-language-scoped.md), [ADR-052](../docs/adr/ADR-052-nnagent-structural-architecture.md)

---

## Architecture

```
nnagent/
├── CMakeLists.txt              # C++ core build config
├── core/                       # C++17 core library (nnagent_core)
│   ├── include/nnagent/        # Public headers
│   └── src/                    # Implementations
├── desktop/                    # Tauri desktop app
│   ├── src-tauri/              # Rust shell + FFI bridge
│   └── web/                    # (symlinked or shared with web/)
├── web/                        # React web UI (shared with desktop/container)
│   ├── src/                    # React + TypeScript source
│   ├── package.json            # npm dependencies
│   └── vite.config.ts          # Vite build config
├── cli/                        # Headless CLI (future)
├── studio_embed/               # QWebEngineView wrapper for Studio (future)
└── tests/
    ├── unit/                   # Google Test unit tests
    └── integration/            # Google Test integration tests
```

### Layer Responsibilities

| Layer | Technology | Framework Dependency |
|-------|------------|---------------------|
| C++ Core | C++17, CMake | None (pure C ABI) |
| Rust Shell | Tauri 2.x, Rust 1.70+ | Tauri (thin adapter) |
| Frontend | React 18, TypeScript 5.x | Tauri IPC (isolated in `tauri-adapter.ts`) |
| CLI | C++17 static | None |

### Escape-Hatch (ADR-050)

The C++ core is 100% framework-agnostic. The Rust shell is a minimal FFI bridge. The React frontend uses `tauri-adapter.ts` as the single point of Tauri dependency, enabling migration to Electron or Qt6 QWebEngineView in 2-4 weeks.

---

## Quick Start

### Prerequisites

- **CMake** 3.20+
- **C++17 compiler** (MSVC 2019+, GCC 11+, Clang 13+)
- **Rust** 1.70+ (for Tauri shell)
- **Node.js** 18+ (for React frontend)
- **System WebView**: WebView2 (Windows), WKWebView (macOS), WebKitGTK 2.34+ (Linux)

### Build C++ Core

```bash
cd nnagent
cmake -B build -S .
cmake --build build --config Release
```

### Run Tests

```bash
cd build
ctest --output-on-failure
```

### Develop Frontend

```bash
cd nnagent/web
npm install
npm run dev
```

### Build Tauri Desktop App

```bash
cd nnagent/desktop/src-tauri
cargo tauri dev
```

---

## Configuration

Per [ADR-052](../docs/adr/ADR-052-nnagent-structural-architecture.md), NNSpire Agent stores all data as flat JSON files in the user's profile folder:

```
~/.nnspire/nnagent/
├── settings.json              # Main app settings
├── providers.json             # Model provider configs
├── profiles.json              # User profiles
├── templates.json             # Prompt templates
├── mcp.json                   # MCP server configs
├── kb.json                    # Knowledge base configs
├── automations/               # Automation templates
├── conversations/             # Chat histories
├── folders.json               # Folder structure index
└── nnagent.log                # Application log
```

---

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| PH1 | 🚧 In Progress | Foundation — project skeleton, config, shell |
| PH2 | 🔒 Locked | Chat Interface |
| PH3 | 🔒 Locked | Model Provider Integration |
| PH4 | 🔒 Locked | Profiles System |
| PH5 | 🔒 Locked | MCP Integration |
| PH6-PH15 | 🔒 Locked | See [`docs/nnagent-TODO.md`](../docs/nnagent-TODO.md) |

---

## Testing

- **C++ Core**: Google Test (unit + integration)
- **TypeScript Frontend**: Vitest + React Testing Library

```bash
# C++ tests
ctest --test-dir build

# TypeScript tests
cd nnagent/web && npm run test
```

---

## License

NNSpire Agent is released under **GPL v3** (consistent with NNSpire Studio pillar).

See the root [`LICENSING.md`](../LICENSING.md) for the full license matrix.
