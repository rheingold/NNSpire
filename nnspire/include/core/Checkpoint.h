/* ============================================================================
 * Checkpoint.h — save/load model weights + optimizer state to .nns files
 * LGPL v3
 *
 * FORMAT (all integers little-endian):
 * ──────────────────────────────────────────────────────────────────────────
 * Header:
 *   magic    : char[4]  = "NNSC"  (NNS Checkpoint)
 *   version  : uint16   = 1
 *
 * Section 1 — Weights:
 *   tag      : char[2]  = "WS"
 *   count    : uint32   (number of parameters)
 *   for each parameter:
 *     name_len : uint32
 *     name     : char[name_len]  (not null-terminated)
 *     tensor   : inline NNS1 binary (same format as Tensor::save)
 *
 * Section 2 — Optimizer state  (written only when state is non-empty):
 *   tag      : char[2]  = "OS"
 *   type_len : uint32
 *   type     : char[type_len]   (optimizer name(), e.g. "Adam")
 *   t        : uint64           (step counter from IOptimizer)
 *   count    : uint32           (number of entries = number of parameters)
 *   for each entry:
 *     has_state : uint8   (0 = no state for this param; 1 = state present)
 *     if has_state:
 *       m_tensor : NNS1 binary
 *       v_tensor : NNS1 binary
 *
 * Section 3 — Training counters:
 *   tag        : char[2]  = "TC"
 *   globalStep : uint64
 *   epoch      : uint64
 *
 * END:
 *   tag      : char[2]  = "EN"
 * ──────────────────────────────────────────────────────────────────────────
 *
 * Notes:
 *  - Raw gradients are intentionally NOT saved.  They are recomputed on
 *    the first backward pass after resume.
 *  - Loading is lenient: unknown section tags are skipped (forward-compat).
 *  - This format is NOT the same as ONNX; ONNX export is a separate system.
 *    Here we aim only for fast, exact resume of a training run.
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
 * ============================================================================ */

#pragma once

#include <core/Result.h>
#include <core/IOptimizer.h>
#include <core/Layer.h>

#include <string>
#include <string_view>
#include <vector>

namespace NNSpire::core {

// ─── CheckpointState — transfer object ───────────────────────────────────────
/**
 * All data captured in / restored from a .nns checkpoint.
 * Pass to Checkpoint::save() / Checkpoint::load().
 */
struct CheckpointState {
    // Model parameters (name + tensor data).
    // Order must be stable — same order as ILayer::parameters() returns them.
    std::vector<Parameter*> params;           ///< must be non-null (write) or pre-allocated (read)

    // Optimizer state (optional — may be nullptr if you only want to save weights).
    IOptimizer* optimizer  {nullptr};

    // Training position counters.
    uint64_t globalStep    {0};
    uint64_t epoch         {0};
};

// ─── Checkpoint ──────────────────────────────────────────────────────────────
/**
 * Stateless utilities for saving and loading .nns checkpoint files.
 *
 * Typical save:
 * @code
 *   CheckpointState cs;
 *   cs.params     = model.parameters();
 *   cs.optimizer  = &adam;
 *   cs.globalStep = trainer.globalStep();
 *   cs.epoch      = currentEpoch;
 *   auto r = Checkpoint::save("run1_epoch5.nns", cs);
 *   if (!r.ok()) { ... }
 * @endcode
 *
 * Typical load:
 * @code
 *   // Model must already be built (parameters allocated) before loading.
 *   CheckpointState cs;
 *   cs.params    = model.parameters();
 *   cs.optimizer = &adam;
 *   auto r = Checkpoint::load("run1_epoch5.nns", cs);
 *   if (!r.ok()) { ... }
 *   uint64_t resumeEpoch = cs.epoch;
 * @endcode
 */
class Checkpoint {
public:
    /**
     * Write a .nns checkpoint to `path`.
     * `cs.params` must be non-empty (the model must be built).
     */
    static Result<void> save(std::string_view path, const CheckpointState& cs);

    /**
     * Read a .nns checkpoint from `path`.
     * On entry `cs.params` must point to the model's already-allocated parameters
     * (same count and order as when the file was saved).
     * On success `cs.globalStep` and `cs.epoch` are updated, and
     * `cs.optimizer` state is populated if the file contains an optimizer section.
     */
    static Result<void> load(std::string_view path, CheckpointState& cs);
};

} // namespace NNSpire::core
