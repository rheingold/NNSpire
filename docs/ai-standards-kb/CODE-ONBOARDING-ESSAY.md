# Code Onboarding, Repository Navigation, and the Documentation Gap

*Written April 2026. Context: NNStudio project discussions.*  
*Two parts: (1) the developer navigation problem; (2) the broader stakeholder and AI-continuity problem.*

---

## Part I — Developer Repository Navigation

### The golden rules exist. You may not know them yet.

There is a well-developed body of practice for reading an unfamiliar codebase quickly.
Almost none of it involves starting from `main()`.
Here is the actual entry sequence that experienced developers use:

**1. `README.md` + `CONTRIBUTING.md` first.**  
Not to understand the software — to understand the *map*.
Every project of any quality tells you what the big pieces are.
If you skip this and go straight to source files, that is a navigation choice with a predictable outcome.

**2. The build system is the dependency graph.**  
`CMakeLists.txt`, `build.gradle`, `Makefile` — whichever it is — tells you which components exist,
which depend on which, and what the entry points are.
This is the architectural diagram that is always up to date because it cannot lie:
if it were wrong, nothing would compile.
Five minutes reading the root build file gives you the component topology faster than reading any source file.

**3. Read the tests before the implementation.**  
A test file for `Trainer` tells you: what `Trainer` is supposed to do, what its interface looks like
from outside, what a complete correct usage looks like.
Tests are the only documentation guaranteed to be correct (when CI is green).
Read `tests/` before `src/`.

**4. Read the interface headers, not the implementation.**  
`ILayer.h` tells you everything a layer must do in 30 lines.
`Dense.cpp` tells you how one specific layer does it in 200 lines of detail.
Start with the former.

**5. `git log --oneline` on a specific file.**  
The commit history of a file is often better documentation than the file itself —
it tells you *why* decisions were made, what changed, and what was tried and reverted.

**6. Search for the domain concept, not the code symbol.**  
If you want to understand how training works, search for `"train"` in README, then in test names,
then in header names. Do not start by searching for a variable.

These six practices are documented and taught — in *Large Scale C++ Software Design* (Lakos),
*Clean Architecture* (Martin), *The Pragmatic Programmer* (Hunt & Thomas) —
but only to people who sought those books out.
If they were not part of your formation, the conventions look arbitrary rather than principled.
That is a gap in education, not a gap in intelligence.

### Is your struggle with bad repos, or with missing technique?

Both, in different proportions. The honest answer:

The golden rules make *well-maintained* projects genuinely fast to navigate. The majority of public
repositories are not well-maintained. Research code pushed once by a PhD student, side projects
with no structural documentation, abandoned prototypes — these exist in volume and they are
genuinely hard to navigate regardless of technique. Your struggle is not incompetence.
It is partly technique and partly the realistic quality distribution of public repositories.

The causes for missing documentation are mixed — not uniformly arrogance:
- More often: the author cannot see the code through the eyes of someone who arrived ten minutes ago
  (the *curse of knowledge* — a known cognitive failure mode, not malice)
- Sometimes: documentation was the first thing cut when time ran short
- Occasionally: genuine "if you can't read it you should not be touching it" gatekeeping
  (exists, particularly in older academic and Unix-culture codebases)
- Most commonly: nobody ever demanded it, so nobody ever developed the practice

---

## Part II — Stakeholders Who Are Not Developers, and the AI-Continuity Problem

### The problem stated precisely

There is a class of roles — product owners, project managers, solution architects, enterprise architects,
business consultants, technical writers, compliance officers — whose job requires genuine understanding
of what a software system does, how it is structured, and why it was built that way,
but whose *daily work* is not reading or writing C++ (or Python, or Java, or whatever the code is in).

The standard responses to their documentation needs are, in order of how often they are given:

1. **"The devs should write proper documentation."**  
   Partially correct but insufficient: devs writing about code they know by heart produce documentation
   that is correct but pitched at the wrong level — it explains *how* without explaining *why* or *what*
   in terms a non-developer can act on. Also, developer time is expensive and writing is slow for people
   who think in code.

2. **"They should ask the devs."**  
   Not repeatable. Not available 24/7. Requires the dev to interrupt their work.
   Produces knowledge that evaporates the moment the meeting ends.
   Does not scale past four people.

3. **"They should learn to read code."**  
   Denies specialisation. A product owner learning to read C++ in order to do their job
   is a product owner no longer doing their job.
   Also creates a security and access-control problem:
   not every stakeholder role should have read access to the entire source tree.

All three answers are real answers in practice. All three are genuinely inadequate in isolation.

### The AI dimension

AI code assistants add a new and underappreciated failure mode to this picture.

When an AI generates code — including repository structure, architectural decisions,
inline comments, and documentation — it can do so faster and more verbosely than any human developer.
The result *looks* well-documented. But:

- The AI's "understanding" is reconstructed on demand from the text in its context window.
  It does not hold state between sessions.
- If the AI that generated the code is no longer available (model deprecated, subscription ended,
  company pivoted), the only entity that could rapidly traverse the entire codebase line-by-line
  to answer "why is this here?" is gone.
- Human team members who relied on the AI as their live documentation system are now in the same
  position as someone who relied on one senior developer who just left: the knowledge exists
  somewhere in the artefacts, but the retrieval mechanism is gone.

This is not a hypothetical risk. It is the natural endpoint of "AI writes code,
AI explains code on demand, humans stop maintaining independent mental models."

Human QA of AI-generated code has the same problem: it requires a human who can read the code
and understand it well enough to recognise when the AI got it subtly wrong.
If the humans on the team cannot do that independently of the AI, the QA is the AI checking itself.

### The one-for-all solution that comes closest

No single practice fully solves this. But one practice comes closer than anything else
to serving all audiences (developers, non-technical stakeholders, future AI sessions, human QA)
with the minimum burden on the people who dislike writing documentation (i.e., developers):

**Architecture Decision Records (ADRs).**

An ADR is a short, structured document — typically one to two pages — that records:
- The problem or question being decided
- The context at the time (forces, constraints, options considered)
- The decision made
- The consequences (what becomes easier, what becomes harder, what is accepted as a trade-off)

Key properties that make ADRs uniquely efficient:

| Property | Why it matters |
|---|---|
| Written once, at decision time | Takes 20-30 minutes when the context is fresh; would take hours to reconstruct later |
| Never needs to be updated | An ADR records what was decided *and why, at that moment*. It does not pretend the decision was always correct. Superseded ADRs are marked superseded, not deleted. |
| Answers "why" not "what" | The code answers "what". The test answers "does it work". The ADR answers "why this instead of the alternative" — the question every non-developer and every new developer asks first. |
| Readable without code knowledge | An ADR written well can be read and understood by a product owner, an enterprise architect, or a compliance officer. |
| Stores the institutional memory | When the original developers leave, the ADRs remain. When the AI that "remembers" the decisions is no longer available, the ADRs remain. |

ADRs are the closest available thing to a "one document for all audiences" practice.

### The C4 Model as a complementary layer

For stakeholders who need to *see* the system rather than read about decisions,
the **C4 Model** (Simon Brown) provides four levels of abstraction mapped exactly to the audience spectrum:

| Level | Audience | What it shows | Requires code knowledge? |
|---|---|---|---|
| Level 1 — System Context | Executive, enterprise architect | What the system does; what users and external systems it interacts with | No |
| Level 2 — Container | Solution architect, PM | Major technical components (apps, databases, services) and their relationships | Minimal |
| Level 3 — Component | Developer, technical architect | Internal structure of one container | Some |
| Level 4 — Code | Developer only | Class/module level | Yes |

A project with one Level-1 and one Level-2 C4 diagram — maintainable in a single Markdown file
using text-based diagram tools like Mermaid or Structurizr Lite — serves every non-developer
stakeholder adequately. Levels 3 and 4 are optional and developer-facing only.

### The Diátaxis principle (for documentation writers)

When professional documentation is needed, the **Diátaxis framework** (Procida) categorises
all documentation into four types with four different audiences and purposes:

| Type | Answers | Audience |
|---|---|---|
| Tutorial | "How do I get started?" | New user, learning mode |
| How-to guide | "How do I do this specific thing?" | Practitioner with a task |
| Reference | "What exactly does this do?" | Developer checking a detail |
| Explanation | "Why is it this way?" | Anyone needing understanding |

Most projects only write Reference (because it is the easiest for developers to produce).
They skip Explanation (which is what ADRs + `CONTRIBUTING.md` provide) and Tutorial
(which is what a `QUICK-START.md` provides). The absence of Explanation is exactly
what makes projects hard to navigate — for developers and non-developers alike.

### Practical minimum for a project like NNStudio

With the constraint of minimum developer burden and maximum coverage:

| Artefact | Effort | Serves |
|---|---|---|
| `README.md` (root) | 1-2 hours once | All audiences: what the software is |
| `CONTRIBUTING.md` (root) | 2-3 hours once, update rarely | Developers, architects: how to navigate the repo |
| `CONTRIBUTING.md` (per folder) | 15-20 mins per folder, written once | Developers: why this folder exists here |
| `blueprints.md` | Ongoing, written alongside code | Developers + non-developers willing to read |
| ADR folder (`decisions/`) | 20-30 mins per significant decision | All audiences: why it is built this way |
| C4 Level 1 + Level 2 diagram | 1-2 hours once, update on major change | All non-developer stakeholders |

The ADR folder is the one most frequently omitted and the one with the highest return per minute of writing time. It also directly solves the AI-continuity problem: any future session — human or AI — that reads the ADR for "why `backends/` is a sibling of `core/`" gets the CUDA-linking argument immediately, without needing to reconstruct it from the code or ask the original developer.

### Human QA and continuity without the AI

The practical recommendation:

> Any architectural decision that was made *with* AI assistance should be recorded in an ADR
> *before* that AI session ends — while the context is present to write it in 20 minutes.
> If the AI is gone tomorrow and the ADR was not written, the decision is effectively undocumented.

NNStudio already does a version of this: `blueprints.md` serves the explanation role,
`TODO.md` tracks the forward decisions, and the `CONTRIBUTING.md` files record the structural rationale.
The missing piece is a formal `decisions/` folder with numbered ADRs for the major
architectural choices (namespace tier design, `backends/`-as-sibling, Plugin SDK ABI plan, etc.).
Those decisions are currently scattered across `blueprints.md` — useful, but not discoverable
by someone who needs to answer one specific "why" question without reading the whole document.

---

## Summary

| Question | Answer |
|---|---|
| Is it your incompetence? | Partially technique; partially genuinely bad repos. Both are real. |
| Do golden rules exist? | Yes. Build file → tests → interface headers → ADRs → then implementation. |
| Why do most repos lack onboarding docs? | Curse of knowledge + nobody demanded it + documentation cut first under time pressure. |
| What serves all audiences with least dev burden? | ADRs (decisions/) + C4 Level 1-2 + per-folder CONTRIBUTING.md. |
| How do you survive losing the AI? | Write ADRs at decision time, before the session ends. The code is the "what"; the ADR is the "why". Without the "why", neither humans nor a fresh AI session can maintain the system confidently. |
