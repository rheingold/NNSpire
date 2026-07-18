# NNSpire Agent PH2: Chat Interface — Continuation Prompt

## Copy This Prompt for New Task

---

**TASK: Implement NNSpire Agent (nnagent) Phase 2 — Chat Interface**

### PROJECT CONTEXT

You are continuing work on the NNSpire Agent (nnagent) project, a Conversational AI Workbench built on Tauri 2.x with a C++17 core library and React/TypeScript frontend.

**Repository:** `d:/plachy/Dokumenty/Dev/AI/NNSpire`  
**nnagent Location:** `nnagent/` subdirectory  
**Current Status:** PH1 Foundation COMPLETE, ready for PH2 implementation

### PH1 COMPLETION STATUS

| Component | Status | Tests |
|-----------|--------|-------|
| C++ Core Library (`libnnagent_core`) | ✅ COMPLETE | 48 unit + 5 integration = 53/53 passing |
| Configuration System (settings/providers/profiles/MCP) | ✅ COMPLETE | Included above |
| Web Frontend (React + Vite) | ✅ COMPLETE | 1/1 Vitest passing |
| Build System (CMake + pnpm) | ✅ COMPLETE | Verified |
| Documentation Updates | ✅ COMPLETE | `docs/nnagent-TODO.md` updated |

**Total Tests Passing: 54/54** (53 C++ + 1 TypeScript)

### BUILD/TEST ENVIRONMENT

- **OS:** Windows 11
- **C++ Compiler:** MSYS2 MinGW-w64 GCC 15.2.0 (located at `C:\msys64\mingw64\bin\`)
- **CMake:** 3.30+ with Ninja generator
- **Node.js:** v22.21.0
- **Package Manager:** pnpm v11.14.0 (CRITICAL: npm v10.9.4 is incompatible with Node.js v22)
- **PowerShell:** Command chaining uses `;` (NOT `&&`)
- **Build Commands:**
  ```powershell
  # C++ Build
  C:\msys64\mingw64\bin\cmake.exe --preset windows-x64-release
  C:\msys64\mingw64\bin\cmake.exe --build --preset windows-x64-release
  C:\msys64\mingw64\bin\ctest.exe --preset windows-x64-release
  
  # Web Build
  cd nnagent/web ; pnpm install
  cd nnagent/web ; pnpm test
  ```

### CODE RULES (MANDATORY)

**code-rule-1: Error Handling**
- ALWAYS assess probability and possible failure modes
- Provide proper error handling with detailed context (NOT generic "XML parsing error" but "Parsing providers.json at line 42: missing required field 'name'")
- Log/rethrow exceptions with business context
- Never report success on failed operations

**code-rule-2: Testing Requirements**
- 1. Write tests for ALL alterations (unit tests for logic, integration tests for I/O)
- 2. RUN the test before considering feature complete
- Tests must be comprehensive (NOT just smoke tests)

### ARCHITECTURE REFERENCES

Read these files BEFORE starting implementation:
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System design overview
- [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) — Command/action/parameter conventions
- [`docs/nnagent-TODO.md`](docs/nnagent-TODO.md) — Full task breakdown
- [`nnagent/CMakePresets.json`](nnagent/CMakePresets.json) — Build configuration
- [`nnagent/core/include/nnagent/`](nnagent/core/include/nnagent/) — Existing C++ headers
- [`nnagent/web/src/`](nnagent/web/src/) — Existing frontend code

### PH2 IMPLEMENTATION TARGETS

Implement the following tasks in order:

#### PH2-1: Basic Chat Window
- [ ] Implement chat input box (textarea with send button)
- [ ] Implement chat message display area
- [ ] Message bubbles with user/AI visual distinction
- [ ] Timestamp display on messages
- [ ] Collapsible/expandable message blocks
- [ ] Code block rendering with syntax highlighting
- [ ] Copy-to-clipboard for code blocks
- [ ] Tests: message send/display functionality

#### PH2-2: Chat History Management
- [ ] Implement conversation list sidebar
- [ ] Folder/category organization support
- [ ] Multi-level folder nesting
- [ ] Auto-generate conversation titles from first prompt
- [ ] Rename conversations/folders functionality
- [ ] Delete conversations/folders functionality
- [ ] Ctrl+C/X/V keyboard shortcuts for conversations
- [ ] Drag-and-drop reorganization
- [ ] Tests: CRUD operations on conversations

#### PH2-3: Chat Export/Import
- [ ] Export single conversation
- [ ] Export folder of conversations
- [ ] Export entire conversation list
- [ ] Import conversations functionality
- [ ] ChatGPT JSON import compatibility
- [ ] Export formats: JSON, Markdown, PDF
- [ ] Tests: round-trip import/export verification

#### PH2-4: Advanced Chat Features
- [ ] Thinking block display (collapsible)
- [ ] MCP call block display
- [ ] Raw JSON view toggle
- [ ] Error reporting with expandable details
- [ ] Customizable user/AI icons
- [ ] Model switching within conversation
- [ ] Tests: thinking blocks, error handling

### IMPLEMENTATION NOTES

1. **Frontend Focus:** PH2 is primarily a frontend implementation (React components) with minimal C++ core changes
2. **Data Persistence:** Use the existing ConfigLoader pattern for storing chat history to JSON files
3. **Tauri Integration:** Use `tauri-adapter.ts` for IPC communication with backend
4. **Testing Strategy:**
   - React component tests using Vitest + React Testing Library
   - Integration tests for persistence layer
   - Mock AI responses for testing (no actual API calls needed)
5. **UI/UX:** Follow existing CSS patterns in `App.css` and `index.css`
6. **State Management:** Consider using React Context or Zustand for chat state

### DELIVERABLES

For each PH2 sub-phase:
1. Implementation code with proper error handling
2. Unit/integration tests (running and passing)
3. Updated documentation in `docs/nnagent-TODO.md`
4. Build verification (no regressions)

### STARTING INSTRUCTIONS

1. Read the architecture and convention documents listed above
2. Review existing code structure in `nnagent/web/src/`
3. Begin with PH2-1 (Basic Chat Window)
4. Implement incrementally with tests after each feature
5. Verify build and test suite passes before moving to next sub-phase
6. Update TODO.md as you complete tasks

---

**END OF CONTINUATION PROMPT**
