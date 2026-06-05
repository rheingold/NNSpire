/* ============================================================================
 * Trainer.h — training loop utility
 * LGPL v3
 *
 * @see docs/blueprints.md — Chapter 7 for the full guided walkthrough.
 *
 * Trainer is the conductor: it owns nothing and coordinates everything.
 * It holds references to a model (ILayer&), a loss function (ILoss&), and
 * an optimizer (IOptimizer&).  Any of those three can be swapped without
 * touching the others.
 *
 * ── DESIGN GUARD (ADR-022 / deferred ADR) ────────────────────────────────────
 * Before modifying this class, spend a moment checking that the change does
 * NOT close the following future path:
 *
 *   When LibTorchBackend is active, the training loop body (trainStep /
 *   trainEpoch / train) may be delegated to PyTorch Lightning, HuggingFace
 *   Accelerate, or DeepSpeed instead of our own implementation, while Trainer
 *   remains the *interface* (TrainCallbacks, globalStep, shouldStop signal).
 *
 * Concretely: keep trainStep() / trainEpoch() / train() as thin, replaceable
 * units. Avoid embedding multi-GPU assumptions, device-placement logic, or
 * gradient-accumulation state that would be hard to excise later.
 * The decision whether to delegate is deferred — see TODO.md §Backend / Trainer.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * The single atomic unit of learning is trainStep(), which executes the
 * six-step gradient descent cycle in strict order:
 *   1. zeroGrad    — clear accumulated gradients from the previous step
 *   2. forward     — input flows through all layers; each saves lastInput_
 *   3. loss.compute— compare prediction vs target → scalar loss value
 *   4. loss.gradient— dLoss/dPrediction — seed gradient for backprop
 *   5. backward    — gradient flows back through layers in reverse;
 *                    each layer fills W.grad_ / b.grad_ and returns dX
 *   6. optimizer.step — Adam/SGD reads every grad_ and nudges weights
 *
 * trainEpoch() and train() are loops calling trainStep() per batch/epoch.
 * TrainCallbacks let the UI (or tests) observe metrics live without
 * coupling the engine to Qt or any specific logging framework.
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#training-loop
 * ============================================================================ */

#pragma once

#include <core/Layer.h>
#include <core/ILoss.h>
#include <core/IOptimizer.h>

#include <functional>
#include <vector>
#include <cstdint>

namespace NNSpire::internal::training {

// Pull in public-API types that Trainer depends on
using namespace NNSpire::core; // NOLINT(google-build-using-namespace)

// ─── Dataset view ─────────────────────────────────────────────────────────────
struct DataBatch {
    Tensor inputs;
    Tensor targets;
};

using Dataset = std::vector<DataBatch>;

// ─── Training metrics ─────────────────────────────────────────────────────────
struct TrainMetrics {
    float loss{0.0f};
    uint64_t epoch{0};
    uint64_t step{0};
};

// ─── Callbacks ────────────────────────────────────────────────────────────────
struct TrainCallbacks {
    std::function<void(const TrainMetrics&)> onBatchEnd;
    std::function<void(const TrainMetrics&)> onEpochEnd;
    /** Return true to stop training early. */
    std::function<bool(const TrainMetrics&)> shouldStop;
};

// ─── Trainer ──────────────────────────────────────────────────────────────────
class Trainer {
public:
    Trainer(ILayer&      model,
            ILoss&        loss,
            IOptimizer&   optimizer)
        : model_(model), loss_(loss), optimizer_(optimizer) {}

    /**
     * Run one forward+backward+step cycle on a single batch.
     * Gradients are zeroed before forward.
     * @return mean loss for the batch.
     */
    Result<float> trainStep(const Tensor& inputs, const Tensor& targets);

    /**
     * Train for one epoch over the dataset.
     * @return mean epoch loss.
     */
    Result<float> trainEpoch(const Dataset& data,
                             uint64_t epochIdx = 0,
                             const TrainCallbacks& callbacks = {});

    /**
     * Train for multiple epochs.
     */
    Result<void> train(const Dataset& data,
                       uint64_t epochs,
                       const TrainCallbacks& callbacks = {});

    /** Run forward only (eval mode); returns predictions. */
    Result<Tensor> predict(const Tensor& inputs);

    /**
     * Enable/disable EvalTrace recording in the underlying model.
     * Has effect only when the model is a ComputeGraph (or any ILayer that
     * implements the trace hook). All other model types silently ignore it.
     */
    void setTraceMode(bool on) noexcept { model_.setTraceMode(on); }

    uint64_t globalStep() const noexcept { return globalStep_; }

private:
    ILayer&     model_;
    ILoss&      loss_;
    IOptimizer& optimizer_;
    uint64_t                globalStep_{0};
};

} // namespace NNSpire::internal::training
