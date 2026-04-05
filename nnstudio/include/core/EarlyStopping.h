/* ============================================================================
 * EarlyStopping.h — patience-based early stopping helper
 * LGPL v3
 *
 * EarlyStoppingCallback generates a TrainCallbacks::shouldStop predicate that
 * halts training when the epoch loss fails to improve for `patience` epochs.
 *
 * USAGE
 * ──────
 *   EarlyStoppingCallback es(5, 1e-4f);   // patience=5, minDelta=1e-4f
 *
 *   TrainCallbacks cb;
 *   cb.shouldStop = es.predicate();   // attach to TrainCallbacks
 *
 *   trainer.train(dataset, 1000, cb); // will stop early if loss plateaus
 *
 * ALGORITHM
 * ──────────
 *   After each epoch:
 *     if (loss < bestLoss_ - minDelta_):
 *         bestLoss_ = loss
 *         counter_  = 0       ← improvement seen, reset patience
 *     else:
 *         counter_++
 *         if counter_ >= patience_:
 *             return true     ← signal STOP
 *   return false
 *
 * The callback is stateful: call reset() to reuse across multiple train() calls.
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#training-loop
 * ============================================================================ */

#pragma once

#include <core/Trainer.h>
#include <functional>
#include <limits>

namespace nnstudio::core {

class EarlyStoppingCallback {
public:
    /**
     * @param patience  Number of epochs with no improvement before stopping.
     * @param minDelta  Minimum change in loss to qualify as an improvement.
     */
    explicit EarlyStoppingCallback(int patience  = 10,
                                   float minDelta = 1e-4f) noexcept
        : patience_(patience), minDelta_(minDelta) {}

    /** Reset internal state (bestLoss, counter). Call before re-training. */
    void reset() noexcept {
        bestLoss_ = std::numeric_limits<float>::infinity();
        counter_  = 0;
    }

    /**
     * Returns a shouldStop predicate to be assigned to TrainCallbacks::shouldStop.
     * The returned function captures *this by reference — ensure the
     * EarlyStoppingCallback outlives the training loop.
     */
    std::function<bool(const TrainMetrics&)> predicate() noexcept {
        return [this](const TrainMetrics& m) -> bool {
            if (m.loss < bestLoss_ - minDelta_) {
                bestLoss_ = m.loss;
                counter_  = 0;
            } else {
                ++counter_;
                if (counter_ >= patience_) return true;
            }
            return false;
        };
    }

    // Accessors
    int    patience()  const noexcept { return patience_;  }
    float  minDelta()  const noexcept { return minDelta_;  }
    float  bestLoss()  const noexcept { return bestLoss_;  }
    int    counter()   const noexcept { return counter_;   }

private:
    int   patience_;
    float minDelta_;
    float bestLoss_{std::numeric_limits<float>::infinity()};
    int   counter_{0};
};

} // namespace nnstudio::core
