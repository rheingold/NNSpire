// ============================================================================
// sequential-call-chain.cpp
// EXAMPLE FILE — NOT compiled, NOT linked.
//
// A single "annotated spaghetti" trace of how Sequential and ILayer
// collaborate during one forward pass and one backward pass for the
// XOR model:
//
//   layers_[0]  Dense(2→4)          — learns the linear transformation
//   layers_[1]  ReLU                — activation (non-linearity)
//   layers_[2]  Dense(4→1)          — collapses to scalar prediction
//   layers_[3]  Sigmoid             — squashes output into (0, 1)
//
// Read this alongside:
//   - blueprints.md  §Chapter 4  (Sequential design)
//   - blueprints.md  §Chapter 3  (Dense + ILayer)
//   - blueprints.md  §3.8        (ActivationBase hierarchy)
//
// Method signatures are taken verbatim from the real headers.
// Error handling (Result<T> checks) is written out the first time,
// then abbreviated as "// ... (same error relay)" for readability.
// ============================================================================

#include <nnstudio/core/Layer.h>        // ILayer, Sequential, LayerPtr
#include <nnstudio/core/layers/Dense.h> // Dense
#include <nnstudio/core/Activations.h>  // ReLU, Sigmoid

using namespace nnstudio;


// ============================================================================
// PART 1 — Building the model
// ============================================================================
//
// After this block:
//   model.layers_ = [ Dense(2→4), ReLU, Dense(4→1), Sigmoid ]
//   No weight matrices exist yet — build() hasn't run.
// ============================================================================

void example_build() {

    Sequential model;

    //                             in_features, bias, weight_init
    model.add<layers::Dense>    (4, true, WeightInit::GlorotUniform)  // layers_[0]
         .add<activations::ReLU>()                                     // layers_[1]
         .add<layers::Dense>    (1, true, WeightInit::GlorotUniform)  // layers_[2]
         .add<activations::Sigmoid>();                                  // layers_[3]

    // Sequential::build() chains the input shape through every layer so each
    // layer can allocate its weight matrices at the right size.
    //
    //   inputShape = {2}   (two XOR input features)
    //
    //   layer[0] Dense::build({2})    → allocates W[4×2], b[4]  → returns {4}
    //   layer[1] ReLU::build({4})     → no-op (no parameters)   → returns {4}
    //   layer[2] Dense::build({4})    → allocates W[1×4], b[1]  → returns {1}
    //   layer[3] Sigmoid::build({1})  → no-op                   → returns {1}
    //
    auto buildResult = model.build({2});
    // buildResult.value() == Shape{1}   ← output shape of the whole network

} // model and all four layers destroyed here (shared_ptr ref-count → 0)


// ============================================================================
// PART 2 — Forward pass
// ============================================================================
//
// Entry point: Trainer calls model.forward(batch)
//
// The data flows:
//
//   x ──► Dense[0] ──► ReLU[1] ──► Dense[2] ──► Sigmoid[3] ──► prediction
//
//   Each arrow is one call to layer->forward(current).
//   "current" starts as x and is overwritten with each layer's output.
//   Data never branches, never forks — strictly left-to-right.
//
// ============================================================================

void example_forward(const Sequential& model, const Tensor& x) {
    //
    // Sequential::forward() — the real implementation, annotated:
    //
    //   Result<Tensor> Sequential::forward(const Tensor& x) {
    //       Tensor current = x;                        ← start with the input
    //       for (auto& layer : layers_) {              ← iterate [0],[1],[2],[3]
    //           auto r = layer->forward(current);      ← virtual call ↓
    //           if (!r.ok()) return r;                 ← relay any error upward
    //           current = r.value();                   ← output becomes next input
    //       }
    //       return Result<Tensor>(current);            ← final output
    //   }
    //
    // Virtual dispatch table for each layer->forward() call:
    //
    //   layer[0]  → Dense::forward(x)
    //   layer[1]  → ReLU::forward(h1)
    //   layer[2]  → Dense::forward(h2)
    //   layer[3]  → Sigmoid::forward(h3)

    // ── call 1 ── Dense::forward(x)   shape: [batch, 2] → [batch, 4] ─────────
    //
    //   lastInput_ = x                               ← saved for backward (Dense)
    //   y = matmul(x, W_.tensor.transpose()) + b_    ← the actual linear math
    //   return y                                     ← shape [batch, 4]
    //
    Tensor h1 = /* layer[0]->forward(x) */ {};

    // ── call 2 ── ReLU::forward(h1)   shape: [batch, 4] → [batch, 4] ─────────
    //
    //   lastInput_ = h1                              ← saved for backward (ReLU)
    //   y[i] = max(0, h1[i])   elementwise
    //   return y
    //
    Tensor h2 = /* layer[1]->forward(h1) */ {};

    // ── call 3 ── Dense::forward(h2)  shape: [batch, 4] → [batch, 1] ─────────
    //
    //   lastInput_ = h2                              ← saved for backward (Dense)
    //   y = matmul(h2, W_.tensor.transpose()) + b_
    //   return y
    //
    Tensor h3 = /* layer[2]->forward(h2) */ {};

    // ── call 4 ── Sigmoid::forward(h3)  shape: [batch, 1] → [batch, 1] ───────
    //
    //   lastOutput_ = σ(h3)   where σ(z) = 1 / (1 + exp(-z))
    //   NOTE: Sigmoid saves the OUTPUT (not input) — it reuses it in backward
    //   return lastOutput_
    //
    Tensor prediction = /* layer[3]->forward(h3) */ {};

    // After all four calls:
    //   Dense[0].lastInput_  = x      (original batch input)
    //   ReLU[1].lastInput_   = h1     (pre-relu activations)
    //   Dense[2].lastInput_  = h2     (post-relu activations)
    //   Sigmoid[3].lastOutput_ = prediction  (the final σ values)
    //
    // No gradients exist yet. Every weight matrix is unchanged. The forward
    // pass was a "read-only" walk from the network's perspective of parameters.
}


// ============================================================================
// PART 3 — Backward pass
// ============================================================================
//
// Entry point: Trainer calls model.backward(seedGrad)
//
//   seedGrad = loss_.gradient(prediction, target)
//            = dLoss/dPrediction        ← the chain rule starting point
//
// Concrete XOR example — one sample, prediction=0.5, target=1.0, loss=BCE:
//
//   loss_.compute(0.5, 1.0)
//     = -(1.0 * log(0.5) + 0.0 * log(0.5))
//     = -log(0.5) ≈ 0.693              ← the scalar "how wrong"
//
//   loss_.gradient(0.5, 1.0)
//     = (-y/p) + ((1-y)/(1-p))
//     = (-1.0/0.5) + (0/0.5)
//     = -2.0                           ← the seed gradient handed to backward()
//
//   The -2.0 is NOT the error (0.5). It is the slope of the loss curve at
//   this prediction value: "if I nudge the prediction up by ε, the loss
//   changes by -2.0·ε" — i.e. it decreases. The negative sign = push upward.
//   The magnitude 2.0 = how urgently.
//
// The gradient flows:
//
//   seedGrad ◄── Dense[2] ◄── ReLU[1] ◄── Dense[0]
//             ◄── Sigmoid[3]
//    (reading right-to-left in layer order: [3],[2],[1],[0])
//
// Two entirely different things happen in this pass — one per layer type:
//
//   Activations (ReLU, Sigmoid):
//       Use their saved input/output to compute dX, return it. No parameters
//       to update. accumulateGrad() is never called.
//
//   Dense layers:
//       Use their saved input to compute dW and db, call accumulateGrad() on
//       their weight and bias tensors. Also return dX for the layer before them.
//
// ============================================================================

void example_backward(const Sequential& model, const Tensor& seedGrad) {
    //
    // Sequential::backward() — real implementation, annotated:
    //
    //   Result<Tensor> Sequential::backward(const Tensor& gradOut) {
    //       Tensor grad = gradOut;
    //       for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    //           auto r = (*it)->backward(grad);      ← virtual call, reverse order
    //           if (!r.ok()) return r;
    //           grad = r.value();                    ← dX becomes next gradOut
    //       }
    //       return Result<Tensor>(grad);             ← dX w.r.t. network input
    //   }
    //
    // Reverse iteration visits layers: [3],[2],[1],[0]

    // ── call 1 ── Sigmoid::backward(seedGrad)   shape unchanged: [batch, 1] ──
    //
    //   Uses: lastOutput_   (saved during forward)
    //
    //   dσ/dz = σ(1 − σ)   where σ = lastOutput_
    //   dX    = seedGrad ⊙ dσ/dz     (elementwise multiply)
    //   return dX
    //
    //   Concrete numbers (prediction=0.5, target=1.0, seedGrad=-2.0):
    //     σ         = 0.5                (lastOutput_, saved during forward)
    //     dσ/dz     = 0.5 * (1 - 0.5) = 0.25
    //     dX        = -2.0 * 0.25 = -0.5
    //
    //   So dX3 = -0.5
    //
    //   Note: this equals (prediction - target) = 0.5 - 1.0 = -0.5.
    //   That is NOT a coincidence. BCE + Sigmoid is a mathematically matched
    //   pair: their combined derivative always simplifies to (p - y).
    //   The two layers are separate in code; their math was chosen to cancel.
    //
    //   NOTE: NO accumulateGrad() — Sigmoid has no parameters.
    //         It only passes the adjusted gradient back to Dense[2].
    //
    Tensor dX3 = /* layer[3]->backward(seedGrad) */ {};

    // ── call 2 ── Dense::backward(dX3)   shape: [batch, 1] → [batch, 4] ──────
    //
    //   Uses: lastInput_   (= h2, the post-ReLU activations, saved during forward)
    //
    //   dW   = dX3^T  @  lastInput_          ← gradient w.r.t. weights
    //   db   = sum(dX3, dim=0)               ← gradient w.r.t. bias
    //   dX   = dX3    @  W_.tensor           ← gradient w.r.t. the layer's input
    //
    //   Concrete numbers (one neuron, one weight):
    //     lastInput_ (one value of h2) = 0.8    (post-ReLU activation)
    //     dX3                          = -0.5   (from Sigmoid::backward above)
    //     dW  = -0.5 × 0.8  = -0.4   ← stored in W_.tensor.grad_
    //     db  = -0.5         = -0.5   ← stored in b_.tensor.grad_
    //     dX  = -0.5 × w    = -0.5 × 0.3 = -0.15  ← returned backward
    //
    //   Then Adam::step reads dW=-0.4 and updates w (0.3 → 0.301).
    //   See blueprints.md §Chapter 6 for the full Adam numerical trace.
    //
    //   W_.tensor.accumulateGrad(dW)   ← adds dW into W_.tensor.grad_
    //   b_.tensor.accumulateGrad(db)   ← adds db into b_.tensor.grad_
    //   return dX                      ← shape [batch, 4]
    //
    //   After this call, Dense[2]'s weight tensor has grad_ populated.
    //   The optimizer will read W_.tensor.grad() on its next step().
    //
    Tensor dX2 = /* layer[2]->backward(dX3) */ {};

    // ── call 3 ── ReLU::backward(dX2)   shape unchanged: [batch, 4] ──────────
    //
    //   Uses: lastInput_   (= h1, the pre-relu values, saved during forward)
    //
    //   mask[i] = (lastInput_[i] > 0) ? 1.0f : 0.0f   ← the derivative of max(0,x)
    //   dX      = dX2 ⊙ mask                           ← "gate" the gradient
    //   return dX
    //
    //   NOTE: NO accumulateGrad() — ReLU has no parameters.
    //
    Tensor dX1 = /* layer[1]->backward(dX2) */ {};

    // ── call 4 ── Dense::backward(dX1)   shape: [batch, 4] → [batch, 2] ──────
    //
    //   Uses: lastInput_   (= x, the original network input, saved during forward)
    //
    //   dW   = dX1^T  @  lastInput_
    //   db   = sum(dX1, dim=0)
    //   dX   = dX1    @  W_.tensor
    //
    //   W_.tensor.accumulateGrad(dW)
    //   b_.tensor.accumulateGrad(db)
    //   return dX    ← dLoss/dNetworkInput.  Nobody uses this in a plain Sequential,
    //                   but it would be used in a nested or branching graph.
    //
    Tensor dX0 = /* layer[0]->backward(dX1) */ {};

    // After all four calls the gradient state is:
    //
    //   Dense[0].W_.tensor.grad_  populated  ← dLoss / d(each weight in layer 0)
    //   Dense[0].b_.tensor.grad_  populated
    //   Dense[2].W_.tensor.grad_  populated  ← dLoss / d(each weight in layer 2)
    //   Dense[2].b_.tensor.grad_  populated
    //
    //   ReLU[1].lastInput_    still held      ← will be overwritten on next forward
    //   Sigmoid[3].lastOutput_  still held    ← same
    //
    // The Optimizer (Adam / SGD) now calls step(params) — it reads each
    // parameter's .grad() and nudges .data_[] in the downhill direction.
    // Then Trainer calls optimizer_.zeroGrad(params) at the start of the
    // NEXT step, zeroing grad_->data_[i] = 0 on every parameter.
}
