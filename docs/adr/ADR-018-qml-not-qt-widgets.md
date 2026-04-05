# ADR-018 — QML (Qt Quick) for the UI, Not Qt Widgets

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

Qt provides two distinct UI toolkits:

| Toolkit | Rendering | Strengths | Weaknesses |
|---|---|---|---|
| **Qt Widgets** | Native painter / software | Mature, accessible, native look | Limited GPU acceleration; hard to animate; complex custom drawing |
| **Qt Quick / QML** | Scene graph, GPU-accelerated (OpenGL/Vulkan/Metal via RHI) | Fluid animations, data-driven UI, GPU rendering, declarative | Less native feel on some platforms; steeper initial learning curve |

NNStudio requires:
- Real-time animated visualisations: training loss curves updating every batch,
  neuron activation heatmaps, weight magnitude plots.
- Interactive graph editor: drag-and-drop layer topology with animated connections.
- Plugin-contributed UI panels (plugins ship QML components that dock into the Studio).
- Cross-platform visual consistency (not "native look" — the Studio has its own design language).

Qt Widgets cannot GPU-accelerate custom drawing efficiently, and plugin-contributed widgets
are significantly harder to sandbox than QML components.

---

## Decision

The NNStudio UI is built with **QML and Qt Quick**. Qt Widgets are not used in `nnstudio/app/`.

- All panels, dialogs, and custom controls are QML (`.qml` files).
- C++ controllers (`ModelCtrl`, `TrainingCtrl`, `BackendCtrl`, `PluginCtrl`, `HelpCtrl`)
  expose engine state to QML via `Q_PROPERTY`, signals, and `QAbstractListModel`/
  `QAbstractItemModel` subclasses.
- Plugin-contributed UI is also QML: plugins ship a `.qml` component that is loaded
  dynamically by `QQmlEngine` into the panel system. Plugin QML communicates only through
  the controller interface — it has no direct access to the engine C++ objects.
- Qt Quick Controls 2 (Material or Fusion style) provides the base widget set.
  A custom NNStudio style is applied on top for branding.
- `QQuickPaintedItem` or `QSGGeometryNode` is used for the high-frequency visualisation
  panels (loss chart, neuron heatmap) where scene-graph-level drawing gives the necessary
  performance.

---

## Consequences

**Positive**
- GPU-accelerated rendering enables smooth real-time visualisations without manual OpenGL code.
- QML's declarative model makes it easy for contributors to add or modify UI panels.
- Plugin QML sandboxing is cleanly enforced: plugins cannot import engine internals directly.
- Animations and transitions are first-class in QML.

**Negative / constraints**
- Accessibility (screen reader) support in Qt Quick is less mature than Qt Widgets as of 2026.
  Flag for review if accessibility is a hard requirement.
- Platform-native look is not guaranteed; NNStudio will have a consistent but non-native appearance.
- QML debugging tooling (Qt Quick Inspector) must be available in developer builds.

**Follow-on**
- All `.qml` files live under `nnstudio/app/qml/`.
- Controller C++ classes in `nnstudio/app/controllers/`.
- Plugin QML component loading tested in the plugin system integration tests.
- Enable `QML_DISABLE_DISK_CACHE=1` in debug builds for faster QML reload cycle.
