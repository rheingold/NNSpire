/* ============================================================================
 * Trainer.cpp — training loop implementation
 * LGPL v3
 * ============================================================================ */

#include <core/Trainer.h>
#include <cassert>
#include <numeric>

namespace nnstudio::core {

// ---------------------------------------------------------------------------
// WALKTHROUGH — Chapter 4: The Training Loop
//
// trainStep() is the atomic unit of learning.  It implements the standard
// gradient descent loop in 6 steps.  Every epoch calls it once per batch.
//
// The full chain for one XOR sample (e.g. input=[1,0], target=1):
//
//  Step 1: zero_grad — wipe gradients from the previous step.  Without this,
//          gradients would *accumulate* across steps and the network would
//          diverge.  (Intentional accumulation across micro-batches is the
//          only exception, not used here.)
//
//  Step 2: forward — input flows through Dense→ReLU→Dense→Sigmoid.
//          Each layer saves its input (lastInput_) for use in step 5.
//
//  Step 3: loss — compares prediction with target, returns a scalar float.
//          For XOR we use BCE (Binary Cross-Entropy):
//            BCE = -( y*log(p) + (1-y)*log(1-p) )
//          Small loss = prediction is close to target.
//
//  Step 4: loss.gradient — computes dLoss/dPrediction.
//          This is the seed gradient; it starts the backward chain.
//          For BCE: dL/dp = (p - y) / (p*(1-p))   (simplified with Sigmoid)
//
//  Step 5: model.backward — the gradient flows *backwards* through the layers
//          in reverse order: Sigmoid→Dense→ReLU→Dense.
//          Each layer:
//            a) computes its own param gradients (dW, db) and accumulates them
//            b) returns dX to the layer before it
//          At the end every Parameter.grad_ holds dLoss/dParam.
//
//  Step 6: optimizer.step — Adam reads grad() from every parameter and
//          applies the update rule with momentum correction:
//            m = β1*m + (1-β1)*g          (1st moment — mean of gradients)
//            v = β2*v + (1-β2)*g²         (2nd moment — variance of gradients)
//            θ = θ - lr * m̂ / (√v̂ + ε)   (bias-corrected update)
//          Weights are now slightly nudged toward lower loss.
// ---------------------------------------------------------------------------
Result<float> Trainer::trainStep(const Tensor& inputs, const Tensor& targets) {
    model_.train();
    auto params = model_.parameters();

    // Step 1: clear gradients from the previous step
    optimizer_.zeroGrad(params);

    // Step 2: forward pass — each layer records its input for backward
    auto predR = model_.forward(inputs);
    if (!predR.ok()) return Result<float>(predR.error());
    const Tensor& pred = predR.value();

    // Step 3: scalar loss (e.g. BCE = -mean(y*log(p) + (1-y)*log(1-p)))
    auto lossR = loss_.compute(pred, targets);
    if (!lossR.ok()) return Result<float>(lossR.error());
    const float lossVal = lossR.value().flat(0);

    // Step 4: seed gradient — dLoss/dPrediction from the loss function
    auto gradR = loss_.gradient(pred, targets);
    if (!gradR.ok()) return Result<float>(gradR.error());

    // Step 5: backward — gradient flows in reverse through all layers;
    //         each layer accumulates dW/db into its parameters and returns dX
    auto backR = model_.backward(gradR.value());
    if (!backR.ok()) return Result<float>(backR.error());

    // Step 6: optimizer updates weights using the accumulated gradients
    optimizer_.step(params);

    ++globalStep_;
    return Result<float>(lossVal);
}

Result<float> Trainer::trainEpoch(const Dataset& data,
                                   uint64_t epochIdx,
                                   const TrainCallbacks& callbacks) {
    float totalLoss = 0.0f;
    uint64_t batches = 0;

    for (const auto& batch : data) {
        auto r = trainStep(batch.inputs, batch.targets);
        if (!r.ok()) return r;
        totalLoss += r.value();
        ++batches;

        if (callbacks.onBatchEnd) {
            TrainMetrics m;
            m.loss  = r.value();
            m.epoch = epochIdx;
            m.step  = globalStep_;
            callbacks.onBatchEnd(m);
        }
        if (callbacks.shouldStop) {
            TrainMetrics m;
            m.loss  = totalLoss / static_cast<float>(batches);
            m.epoch = epochIdx;
            m.step  = globalStep_;
            if (callbacks.shouldStop(m))
                break;
        }
    }

    const float meanLoss = batches > 0 ? totalLoss / static_cast<float>(batches) : 0.0f;
    if (callbacks.onEpochEnd) {
        TrainMetrics m;
        m.loss  = meanLoss;
        m.epoch = epochIdx;
        m.step  = globalStep_;
        callbacks.onEpochEnd(m);
    }
    return Result<float>(meanLoss);
}

Result<void> Trainer::train(const Dataset& data,
                             uint64_t epochs,
                             const TrainCallbacks& callbacks) {
    for (uint64_t e = 0; e < epochs; ++e) {
        auto r = trainEpoch(data, e, callbacks);
        if (!r.ok()) return Result<void>(r.error());
    }
    return Result<void>();
}

Result<Tensor> Trainer::predict(const Tensor& inputs) {
    model_.eval();
    return model_.forward(inputs);
}

} // namespace nnstudio::core
