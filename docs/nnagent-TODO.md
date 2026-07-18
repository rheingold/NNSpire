# NNSpire Agent (`nnagent`) — Implementation TODO

> **Created:** 2026-07-16
> **Updated:** 2026-07-18 (PH2-1 Basic Chat Interface completed, 27 tests passing)
> **Source:** [`docs/nnagent-requirements-summary.md`](docs/nnagent-requirements-summary.md)
> **Status:** ✅ **PH1 Foundation COMPLETE** — ✅ **PH2-1 Chat Interface COMPLETE** — PH2-2 in progress
> **ADRs:** [`ADR-050`](adr/ADR-050-nnagent-ui-framework-tauri.md), [`ADR-051`](adr/ADR-051-nnagent-implementation-language-scoped.md), [`ADR-052`](adr/ADR-052-nnagent-structural-architecture.md)

---

## PREREQUISITE: Architectural Decisions (BLOCKERS) — ✅ RESOLVED

> ✅ **All 15 decisions confirmed on 2026-07-16. See ADR-050 through ADR-052.**

### P0-1: Platform/Framework Decision ✅
- [x] **Tauri 2.x** (Rust shell + React/TypeScript frontend) — ADR-050
  - [x] Escape-hatch architecture (C++ core framework-agnostic)
  - [x] React web UI for container deployment
- [x] Target platform priority: Desktop → CLI → Service → Container → Mobile

### P0-2: Language/Runtime Decision ✅
- [x] **C++17** for core library (ADR-051)
- [x] **TypeScript** for frontend (ADR-050)
- [x] **Rust** for Tauri shell (ADR-050)
- [x] **Python** for automation scripts + plugin interfaces only (ADR-051)
- [x] Engine integration: HTTP/REST + IPC (ADR-052)

### P0-3: Project Structure Decision ✅
- [x] `nnagent/` at repo root (ADR-052)
- [x] Mixed build: CMake (core/cli) + Cargo (desktop) + npm (web)

### P0-4: Data Storage Decision ✅
- [x] **JSON files** primary (single-user, ADR-052)
- [x] **SQLite** only for multi-user mode (ADR-052)
- [x] Per-conversation files in folder structure (ADR-052)

### P0-5: Plugin Architecture Decision ✅
- [x] Native DLL/SO loading (C++) + Python .pyd (ADR-051)
- [x] Soft PKI with sandboxing for unsigned plugins (ADR-052)
- [x] Reuse NNSpire PKI infrastructure (ADR-007)

---

## PHASE 1: Foundation & Core Infrastructure — ✅ COMPLETE (2026-07-18)

### PH1-1: Project Skeleton ✅
- [x] Create `nnagent/` directory structure
- [x] Set up build system configuration (CMakeLists.txt, CMakePresets.json)
- [x] Create `README.md` for the pillar
- [x] Set up CI/CD pipeline configuration (CMakePresets for local builds)
- [x] Configure linting/formatting tools (ESLint, Prettier for web)
- [x] Create initial test framework (Google Test + Vitest)

### PH1-2: Configuration System ✅
- [x] Design configuration file schema (JSON) — [`ConfigValidator.h`](nnagent/core/include/nnagent/config/ConfigValidator.h)
- [x] Implement configuration loader — [`ConfigLoader.cpp`](nnagent/core/src/config/ConfigLoader.cpp)
- [x] Implement configuration validator — [`ConfigValidator.cpp`](nnagent/core/src/config/ConfigValidator.cpp)
- [x] Create settings UI component — [`App.tsx`](nnagent/web/src/App.tsx)
- [x] Support for multiple configuration profiles (profiles.json schema)
- [x] Configuration export/import functionality (JSON save/load)
- [x] Tests: configuration load/save/validate (10 unit + 5 integration tests)

### PH1-3: Core Application Shell ✅
- [x] Implement main application window — Tauri conf [`tauri.conf.json`](nnagent/desktop/src-tauri/tauri.conf.json)
- [x] Implement basic menu bar — React Router nav in [`App.tsx`](nnagent/web/src/App.tsx)
- [x] Implement settings dialog — Settings route in [`App.tsx`](nnagent/web/src/App.tsx)
- [x] Implement about dialog — About route in [`App.tsx`](nnagent/web/src/App.tsx)
- [x] Application lifecycle management (start/stop/cleanup) — [`AppShell.cpp`](nnagent/core/src/core/AppShell.cpp)
- [x] Logging infrastructure — [`Logger.cpp`](nnagent/core/src/logging/Logger.cpp)
- [x] Tests: application startup/shutdown (10 AppShell + 7 Logger unit tests)

---

### PH1 Test Results Summary (2026-07-18)
| Test Suite | Tests | Status |
|------------|-------|--------|
| C++ Unit Tests (AppShell) | 10 | ✅ PASS |
| C++ Unit Tests (Logger) | 7 | ✅ PASS |
| C++ Unit Tests (ConfigValidator) | 21 | ✅ PASS |
| C++ Unit Tests (ConfigLoader) | 10 | ✅ PASS |
| C++ Integration Tests | 5 | ✅ PASS |
| TypeScript Vitest (App.test.tsx) | 1 | ✅ PASS |
| **TOTAL** | **54** | **✅ ALL PASS** |

### PH1 Bugs Fixed During Implementation
1. **Logger singleton state pollution** — Added `LoggerTest` fixture with SetUp/TearDown for test isolation
2. **File handle leak in AppShell::shutdown()** — Added `set_file_output(false)` to release log file handle
3. **ConfigValidator field name mismatch** — Fixed `require_string` to use actual JSON keys, not prefixed names
4. **Logger callback deadlock** — Moved callback invocation outside mutex lock in `write()`
5. **Empty JSON object vs null** — Fixed `validate_settings` and `validate_mcp` to accept `nlohmann::json::object()`
6. **Web test ambiguity** — Changed `getByText()` to `getByRole('heading')` for unique element matching

---

## PHASE 2: Chat Interface

### PH2-1: Basic Chat Window ✅ COMPLETE (2026-07-18)
- [x] Implement ChatContext with reducer for state management
- [x] Implement chat input box (auto-resizing textarea)
- [x] Implement chat message display with scroll
- [x] Message bubbles with user/AI distinction + role icons
- [x] Timestamp display
- [x] Collapsible/expandable message blocks (thinking, MCP, error)
- [x] Code block rendering with syntax highlighting (highlight.js)
- [x] Copy-to-clipboard for code blocks
- [x] ChatWindow with mock AI responses for testing
- [x] localStorage persistence with Tauri backend fallback
- [x] Tests: 27 tests passing (chat.test.ts + ChatContext.test.tsx + App.test.tsx)
- [x] Build: `pnpm run build` passes, `dist/` produced
- [ ] Windows distributable: Requires Rust/Cargo toolchain (not in current env)

### PH2-2: Chat History Management
- [ ] Implement conversation list sidebar
- [ ] Folder/category organization
- [ ] Multi-level folder support
- [ ] Auto-generate conversation titles from first prompt
- [ ] Rename conversations/folders
- [ ] Delete conversations/folders
- [ ] Ctrl+C/X/V for conversations
- [ ] Drag-and-drop reorganization
- [ ] Tests: CRUD operations on conversations

### PH2-3: Chat Export/Import — ✅ COMPLETE (2026-07-18)
- [x] Export single conversation — ✅ exportConversation in ChatContext
- [x] Export folder of conversations — ✅ exportFolder
- [x] Export entire conversation list — ✅ exportAll
- [x] Import conversations — ✅ importFromFile with auto-detect
- [x] ChatGPT JSON import compatibility — ✅ chatGptImporter.ts
- [x] Export formats: JSON, Markdown, PDF — ✅ exportFormatter.ts
- [x] UI: Import button (📥) in sidebar, export (📤) in context menu
- [x] Tests: exportFormatter (9), chatGptImporter (12), round-trip in ChatContext + Sidebar

### PH2-4: Advanced Chat Features — ✅ COMPLETE (2026-07-18)
- [x] Thinking block display (collapsible) — ✅ in ChatMessage component
- [x] MCP call block display — ✅ in ChatMessage component
- [x] Raw JSON view toggle — ✅ in ChatMessage component
- [x] Error reporting with expandable details — ✅ in ChatMessage component
- [x] Customizable user/AI icons (theming) — ✅ ChatTheme + setTheme in ChatContext, icon palette in ChatWindow
- [x] Model switching within conversation — ✅ SET_MODEL reducer + model selector in ChatWindow
- [x] Tests: theming (5 tests), model switching (4 tests), ChatTheme type (2 tests) — ✅ ChatContext.theming.test.tsx

---

## PHASE 3: Model Provider Integration

### PH3-1: Provider Abstraction Layer
- [ ] Define provider interface/contract
- [ ] Implement provider registry
- [ ] Provider configuration UI
- [ ] Tests: mock provider implementation

### PH3-2: OpenAI-Compatible Provider
- [ ] Implement OpenAI API client
- [ ] Support for chat completions
- [ ] Support for streaming responses
- [ ] Model listing from API
- [ ] Authentication (API keys)
- [ ] Tests: integration with OpenAI API

### PH3-3: Additional Providers
- [ ] Anthropic provider
- [ ] LM Studio provider
- [ ] Ollama provider (local models)
- [ ] Custom provider support
- [ ] Tests: each provider

### PH3-4: Model Configuration
- [ ] Per-model default parameters (temperature, etc.)
- [ ] System prompt configuration
- [ ] Model favorites system
- [ ] Default model selection
- [ ] Tests: model parameter application

---

## PHASE 4: Profiles System

### PH4-1: Profile Data Model
- [ ] Define profile schema
- [ ] Profile storage implementation
- [ ] Profile CRUD operations
- [ ] Tests: profile management

### PH4-2: Profile UI
- [ ] Profile selection dropdown in chat
- [ ] Profile editor dialog
- [ ] Profile favorites
- [ ] Default profile configuration
- [ ] Model filtering based on profile
- [ ] Tests: profile selection/application

### PH4-3: Profile-Model Interactions
- [ ] Profile forces model switch
- [ ] Model selection filters profiles
- [ ] Default model + default profile interaction
- [ ] Tests: bidirectional filtering

---

## PHASE 5: MCP Integration

### PH5-1: MCP Client Foundation
- [ ] Implement MCP client
- [ ] MCP server discovery
- [ ] Tool listing from MCP servers
- [ ] Tests: MCP connection

### PH5-2: Default MCP Tools
- [ ] Web fetch tool
- [ ] File read tool
- [ ] File write tool
- [ ] Terminal/command execution tool
- [ ] Tests: each tool

### PH5-3: MCP Security
- [ ] Firewall configuration for file paths
- [ ] Firewall configuration for commands
- [ ] Default deny/allow policy
- [ ] User permission prompts
- [ ] Permission caching (always/this session)
- [ ] Tests: security policies

---

## PHASE 6: Prompt Templates

### PH6-1: Template Data Model
- [ ] Define template schema
- [ ] Template storage
- [ ] Template CRUD operations
- [ ] Tests: template management

### PH6-2: Template Editor
- [ ] Template creation/editing UI
- [ ] Placeholder syntax support `{{}}`
- [ ] Switchable parts with `*-text-*` syntax
- [ ] Intellisense for placeholders
- [ ] Tests: template editing

### PH6-3: Template Usage
- [ ] Reserved character trigger (`@` or `#`)
- [ ] Template completion in chat input
- [ ] Placeholder value input dialog
- [ ] Switchable part checkboxes
- [ ] Template expansion in chat display
- [ ] Tests: template insertion

### PH6-4: External Template Sources
- [ ] Load templates from files
- [ ] Load templates from folders
- [ ] Multi-user template sharing
- [ ] Tests: external template loading

---

## PHASE 7: Knowledge Base & RAG

### PH7-1: KB Data Model
- [ ] Define KB record schema
- [ ] KB storage implementation
- [ ] KB CRUD operations
- [ ] Tests: KB management

### PH7-2: File Indexing
- [ ] Local file indexing
- [ ] File parser plugin integration
- [ ] Indexing status indicator
- [ ] Manual trigger for reindexing
- [ ] Scheduled reindexing
- [ ] Budget controls (tokens, prompts, size)
- [ ] Tests: file indexing

### PH7-3: RAG Integration
- [ ] Vector DB plugin interface
- [ ] Qdrant plugin (default)
- [ ] Embedding model configuration
- [ ] Document chunking
- [ ] RAG query integration with chat
- [ ] Tests: RAG pipeline

### PH7-4: Advanced KB Sources
- [ ] Database connectors (ODBC, PostgreSQL, etc.)
- [ ] PIM database connectors (Outlook, Thunderbird)
- [ ] Registry source
- [ ] Tests: advanced sources

---

## PHASE 8: File Parsers

### PH8-1: Parser Plugin Architecture
- [ ] Define parser plugin interface
- [ ] Plugin loading mechanism
- [ ] Default plaintext parser
- [ ] Tests: plugin loading

### PH8-2: Default Parsers
- [ ] PDF parser
- [ ] DOCX parser
- [ ] Image parser (OCR)
- [ ] Audio parser (transcription)
- [ ] Tests: each parser

### PH8-3: Parser Integration
- [ ] Parser selection per file type
- [ ] Read-only vs read/write parsers
- [ ] Output block handling in chat
- [ ] Save-as functionality
- [ ] Tests: parser integration

---

## PHASE 9: Automations

### PH9-1: Automation Data Model
- [ ] Define automation template schema
- [ ] Block types definition
- [ ] Connection/link definition
- [ ] Tests: data model

### PH9-2: Automation Editor UI
- [ ] Graph editor canvas
- [ ] Block palette
- [ ] Drag-and-drop block placement
- [ ] Connection drawing between blocks
- [ ] Block property editors
- [ ] Tests: editor interactions

### PH9-3: Block Types Implementation
- [ ] AI Prompt block
- [ ] Script block (JavaScript/Python)
- [ ] DLL/Plugin block
- [ ] Timer block
- [ ] Decision block
- [ ] Group block
- [ ] Tests: each block type

### PH9-4: Automation Runner
- [ ] Execute automation from editor
- [ ] Visual execution highlighting
- [ ] Console logging
- [ ] Global parameter injection
- [ ] Parameter passing between blocks
- [ ] Tests: automation execution

### PH9-5: Automation Scheduling
- [ ] Timer-based scheduling
- [ ] Automation instances
- [ ] Ready-to-run automation list
- [ ] Background worker support
- [ ] Tests: scheduling

---

## PHASE 10: Multi-User & Authorization

### PH10-1: User Management
- [ ] User data model
- [ ] User CRUD operations
- [ ] Authentication system
- [ ] Tests: user management

### PH10-2: Group Management
- [ ] Group data model
- [ ] Group CRUD operations
- [ ] User-group assignments
- [ ] Tests: group management

### PH10-3: Authorization System
- [ ] Permission model
- [ ] Role-based access control
- [ ] Per-resource authorization
- [ ] Authorization checks in UI
- [ ] Tests: authorization enforcement

### PH10-4: SSO Integration
- [ ] OAuth support
- [ ] LDAP/Active Directory support
- [ ] Windows integrated auth
- [ ] Tests: SSO flows

---

## PHASE 11: API & Remote Access

### PH11-1: REST API Foundation
- [ ] API server implementation
- [ ] Authentication middleware
- [ ] Rate limiting
- [ ] Tests: API endpoints

### PH11-2: Chat API
- [ ] Send message endpoint
- [ ] List conversations endpoint
- [ ] Get conversation endpoint
- [ ] Streaming response support
- [ ] Tests: chat API

### PH11-3: Model Router API
- [ ] OpenAI-compatible proxy endpoint
- [ ] Model listing endpoint
- [ ] Provider routing
- [ ] Tests: model routing

### PH11-4: Automation API
- [ ] Trigger automation endpoint
- [ ] List automations endpoint
- [ ] Get automation status endpoint
- [ ] Tests: automation API

---

## PHASE 12: Deployment & Packaging

### PH12-1: Desktop Packaging
- [ ] Windows installer
- [ ] macOS DMG
- [ ] Linux AppImage/Deb
- [ ] Auto-update mechanism
- [ ] Tests: installation

### PH12-2: Container Deployment
- [ ] Dockerfile
- [ ] Docker Compose configuration
- [ ] Web UI in container
- [ ] Health checks
- [ ] Tests: container deployment

### PH12-3: Service/Daemon Mode
- [ ] Windows Service installer
- [ ] systemd service file
- [ ] macOS launchd plist
- [ ] Configuration for headless mode
- [ ] Tests: service mode

### PH12-4: Mobile Packaging
- [ ] iOS app packaging
- [ ] Android APK/AAB
- [ ] Widget/Today Screen integration
- [ ] Tests: mobile deployment

---

## PHASE 13: Theming & Skinning

### PH13-1: Theme System
- [ ] Theme descriptor schema
- [ ] Built-in themes (light, dark, etc.)
- [ ] Theme hot-reload
- [ ] Tests: theme switching

### PH13-2: Skin System
- [ ] Skin descriptor schema
- [ ] Layout customization
- [ ] Default skins per platform
- [ ] Tests: skin application

### PH13-3: Responsive Design
- [ ] Mobile layout
- [ ] Hamburger menu
- [ ] Bottom control bar
- [ ] Tests: responsive layouts

---

## PHASE 14: Cross-Pillar Integration

### PH14-1: NNSpire Studio Integration
- [ ] Embedded Agent Panel
- [ ] MCP boundary node implementation
- [ ] Bidirectional communication
- [ ] Tests: studio integration

### PH14-2: NNSpire Engine Integration
- [ ] Runner API client
- [ ] NMID file reader
- [ ] Raw NN input support
- [ ] Tests: engine integration

---

## PHASE 15: Profile Switching & Environment

### PH15-1: Environment Profiles
- [ ] Local storage profile
- [ ] Database connection profile
- [ ] Remote API profile
- [ ] Profile switching UI
- [ ] Tests: profile switching

### PH15-2: Remote Folder Support
- [ ] NFS/FTP/SMB/CIFS support
- [ ] WebDAV/SharePoint/OneDrive/Google Drive
- [ ] Scheduled sync
- [ ] On-demand load
- [ ] Tests: remote folders

---

## TESTING REQUIREMENTS

> Per project conventions (code-rule-2), every feature requires:

### Unit Tests
- [ ] All data models
- [ ] All business logic
- [ ] All utility functions

### Integration Tests
- [ ] API endpoints
- [ ] Provider integrations
- [ ] MCP tool execution
- [ ] File parser pipeline
- [ ] RAG pipeline

### End-to-End Tests
- [ ] Chat workflow
- [ ] Automation execution
- [ ] Profile switching
- [ ] Theme switching

---

## DEPENDENCIES

| Task | Depends On |
|------|-----------|
| PH1-* | P0-1 to P0-5 |
| PH2-* | PH1-* |
| PH3-* | PH1-* |
| PH4-* | PH3-* |
| PH5-* | PH1-* |
| PH6-* | PH1-* |
| PH7-* | PH5-*, PH8-* |
| PH8-* | PH1-* |
| PH9-* | PH2-*, PH3-*, PH5-* |
| PH10-* | PH1-* |
| PH11-* | PH2-*, PH3-*, PH9-* |
| PH12-* | All previous |
| PH13-* | PH1-* |
| PH14-* | PH5-* |
| PH15-* | PH1-* |

---

## NOTES

1. **Each TODO item can be implemented independently** once the prerequisites are resolved
2. **Testing is mandatory** for each item before marking complete
3. **Error handling** must follow code-rule-1 (detailed error messages with context)
4. **Architect must approve** completion of each Phase before proceeding to next
