# ADR-050 — Tauri with Escape-Hatch for NNSpire Agent UI Framework

**Date:** 2026-07-16
**Status:** Accepted
**Decided by:** Project founder (architectural discussion session)
**Supersedes:** Decision point P0-1 in [`docs/nnagent-TODO.md`](docs/nnagent-TODO.md)
**Related:** [`ADR-001`](ADR-001-qt6-ui-framework.md) (Studio remains Qt6), [`ADR-004`](ADR-004-dual-language-everywhere.md), [`ADR-018`](ADR-018-qml-not-qt-widgets.md)

---

## Context

NNSpire Agent (`nnagent/`) requires a UI framework supporting: Desktop (Win/Mac/Linux), Web (container), Mobile (Android/iOS), CLI, and Service/Daemon modes. NNSpire Studio (`nnspire/app/`) remains Qt6/QML per ADR-001/ADR-018. Agent must be embeddable in Studio via MCP boundary (ADR-044).

Extensive architectural discussion (2026-07-16) evaluated three candidates:
1. **Qt6/QML** — consistent with Studio, single-process, resource-efficient, but requires separate web/mobile UI codebases
2. **Electron** — unified JS codebase, but 200MB+ binary, 600MB+ RAM, no mobile support
3. **Tauri** — thin Rust shell + system WebView + React/TypeScript frontend, 20-30MB binary, unified desktop/web codebase

---

## Decision

**Tauri 2.x** is selected as the primary UI framework for NNSpire Agent desktop and web deployments, with an **escape-hatch architecture** ensuring bounded migration risk.

### Architecture Summary

```
nnagent/
├── core/                      # C++ static lib (nnagent-core) — framework-agnostic
│   └── extern "C" API exports
├── desktop/                   # Tauri app (Rust shell + React/TS frontend)
│   ├── src-tauri/             # Rust FFI bridge to C++ core
│   └── src/                   # React + TypeScript UI
├── web/                       # React web UI (shared with desktop/src)
├── cli/                       # C++ headless CLI — legacy compatible
├── mobile/                    # Capacitor wrapper (future)
└── studio_embed/              # QWebEngineView wrapper for Studio
```

### Layer Responsibilities

| Layer | Technology | Framework Dependency |
|-------|------------|---------------------|
| C++ Core | C++17, CMake | None (pure C ABI) |
| Rust Shell | Tauri 2.x, Rust 1.70+ | Tauri (thin adapter) |
| Frontend | React 18, TypeScript 5.x | Tauri IPC (isolated in `tauri-adapter.ts`) |
| CLI | C++17 static | None |
| Studio Embed | QML + QWebEngineView | Qt6 (existing) |

### Escape-Hatch Design

1. **C++ core is 100% framework-agnostic** — zero Tauri/Qt/Electron dependencies
2. **Rust shell is minimal** — only FFI bindings, no business logic
3. **React frontend uses adapter pattern** — `tauri-adapter.ts` isolates Tauri-specific APIs
4. **Migration paths predefined:**
   - Tauri → Electron: 2-4 weeks (swap Rust shell for Node.js, same React frontend)
   - Tauri → Qt6 QWebEngineView: 1-2 weeks (same React frontend, different host)
5. **Review date:** 18 months from implementation start — reassess Tauri ecosystem health

---

## Verdict Arguments (Founder's Rationale)

### 1. AI Vibe-Coding Era Reduces Migration Cost

We are in the AI vibe-coding era. Porting from one framework to another is still non-trivial, yet it has become a "snip" compared to previous manual processes. The costs of an eventually necessary framework change have dropped significantly, even for a solo vibe-developer. AI-assisted code transformation can handle the mechanical parts of migration (API surface changes, build system updates), leaving only architectural integration as manual work. This fundamentally changes the risk calculus: framework lock-in is no longer a 6-month migration — it's a 2-week AI-assisted swap.

### 2. Maximax Strategy Appropriate for Uncontracted Project

We are not developing a multi-million contract, not a contracted project, not one that aspires to become a worldwide de-facto standard. The probability of happening so is negligible. A higher risk profile is therefore absolutely OK, especially considering the high probability of working out positively (maximax strategy). The upside — unified codebase across 5 deployment targets, 90% smaller binary than Electron, rapid AI-assisted development — outweighs the bounded downside — 2-4 week migration if Tauri stagnates.

### 3. Vibe-Coding Neutralizes Experience Gap

The fact that the founder cannot program the implementation manually is not a Tauri/Electron/TS/Rust issue — it's the new technology explosion. With 5-10 year old technologies proliferating rapidly, no single developer can maintain 30+ years of equivalent experience across all stacks. Vibe-coding (AI-assisted development) actually saves the day here. The AI handles the framework-specific details; the founder provides architectural direction and validation. Basing the decision on "I'd feel safer with Qt because I know it better" would be dishonest — the AI is the actual implementor, and it is framework-agnostic.

---

## Probability Assessment

| Framework | Survival Probability (to 2028) | Migration Cost if Failed |
|-----------|------------------------------|-------------------------|
| Tauri 2.x | ~65% | 2-4 weeks (to Electron) |
| Qt6 | ~85% | 3-6 months (QML→React if web needed) |
| Electron | ~90% | N/A (already proven) |

**Selected:** Tauri — because bounded downside + AI-assisted migration + maximax upside.

---

## Consequences

### Positive
- Single React/TypeScript codebase for desktop + web UI (~90% code reuse)
- Small binary footprint (20-30 MB vs. Electron's 200 MB)
- System WebView auto-updates (security maintained by OS vendors)
- Mobile deployment path via Capacitor (same React components)
- Studio embedding via QWebEngineView (same React UI, different host)
- AI-assisted development velocity maximized (large TS/React ecosystem)

### Negative / Constraints
- Multi-process model (Rust shell + WebView renderer + C++ core = 3 processes)
- Higher RAM usage than Qt6 (~200 MB active vs. ~100 MB)
- Tauri ecosystem younger — fewer production references
- Linux requires WebKitGTK 2.34+ (Ubuntu 20.04+ minimum)
- Windows 7 support requires WebView2 bootstrapper bundling
- Legacy targets (Win XP, MS-DOS, etc.) are CLI-only regardless

### Follow-on
- ADR-051 will cover implementation language details (C++ core + TypeScript frontend + Python only for automation scripts)
- `nnagent/core/` must maintain strict C ABI boundary — no C++ template exports
- `tauri-adapter.ts` must be the ONLY file importing `@tauri-apps/api`
- CMake configuration must support building `nnagent-core` independently of Tauri
- Review scheduled: 18 months post-PH1 completion

---

## Technical Deep-Reference

### Tauri Process Model

```
nnagent.exe (Rust — ~10 MB RAM)
├── WebView2 Renderer (Chromium — ~120 MB RAM)
│   └── [React App JavaScript runs here]
├── nnagent-core.dll (C++ — ~30 MB RAM)
└── [Node.js automation runtime — spawned on demand]
```

### IPC Flow (Frontend → C++ Core)

```
React invoke('cmd', {data})
  → tauri://command/ IPC (JSON-RPC over named pipe)
    → Rust #[tauri::command] handler
      → FFI: extern "C" call into nnagent-core.dll
        → C++ business logic executes
          → Returns char* (JSON response)
            → Rust deserializes to struct
              → Returns to JS via IPC
```

### Reverse Flow (C++ Core → Frontend Events)

```
C++ core event (streaming response, progress)
  → FFI callback to Rust
    → tauri::AppHandle.emit_all("event--name", payload)
      → Frontend listen('event--name', handler) receives
```

### Memory Budget

| Component | Idle | Active |
|-----------|------|--------|
| Rust backend | 10 MB | 20 MB |
| WebView renderer | 40 MB | 120 MB |
| C++ core | 10 MB | 30 MB |
| OS overhead | 20 MB | 40 MB |
| **Total** | **~80 MB** | **~210 MB** |

### Platform Minimums

| Platform | Minimum | WebView | Notes |
|----------|---------|---------|-------|
| Windows | 10 (1809) | WebView2 | Auto-installs if missing |
| macOS | 11 (Big Sur) | WKWebView | Native |
| Linux | Ubuntu 20.04 | WebKitGTK 2.34+ | Distro-managed |
| Android | 5.0 | AndroidX WebView | Beta support |
| iOS | 12.0 | WKWebView | Beta support |
