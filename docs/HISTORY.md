# A Brief History of Neural Network Activations and Their Biological Origins

> This document traces how each layer type used in NNSpire came into existence —
> whether from direct neuroscience observation, mathematical necessity, or pure engineering.
> At the end, a reference table maps each NNSpire layer/activation to its historical context.

---

## The biologically grounded era (1943–1980s)

The very first artificial neuron — the **McCulloch-Pitts unit (1943)** — was designed by a
neuroscientist (Warren McCulloch) and a logician (Walter Pitts) working directly from biology.
The observation was real and measurable: a biological neuron fires an action potential ("spike")
or it doesn't — all or nothing. Below some membrane potential threshold: silence. Above it: a
full spike. That is a **step function** (Heaviside). The first perceptrons used exactly this.

When backpropagation required *differentiable* activations (late 1980s), the step function had to
go — its derivative is zero almost everywhere. **Sigmoid** and **tanh** replaced it. Both happen
to roughly match something real: the **f-I curve** of cortical neurons — the relationship between
input current and output *firing rate* (spikes per second). That curve is S-shaped, saturating
at the maximum firing rate. Sigmoid is a plausible fit to this curve. This was not coincidence —
the early researchers (Rumelhart, Hinton, Williams — the 1986 backpropagation paper) knew the
biology. Sigmoid was the biologically motivated differentiable stand-in for the step function.

---

## ReLU — the interesting dual case

ReLU has partial biological precedent, but it was not adopted *because* of biology.

**Hubel and Wiesel** (Nobel Prize 1981) recorded from individual neurons in the primary visual
cortex of cats and monkeys in the late 1950s–60s. They found "simple cells" that respond only
to edges at specific orientations — and crucially, they respond only to one *polarity* of
contrast. A black-on-white edge at 45° activates the cell; the reverse edge does not. That is
half-wave rectification. That is ReLU.

But when **Glorot et al. (2011)** and **Krizhevsky et al. (AlexNet, 2012)** pushed ReLU into
mainstream deep learning, the reasons were engineering-first:

- Sigmoid's gradient is at most 0.25 and approaches 0 at both ends → vanishing gradients in deep networks
- ReLU's gradient is exactly 1 for all positive inputs → gradients pass through cleanly
- Computationally trivial: `max(0, x)`
- Creates sparse activations (many neurons output exact zero) → less redundancy, faster computation

The biological parallel was noted and cited as supporting context, but the adoption driver was
the **vanishing gradient problem**, not neuroscience papers.

---

## GELU, Swish, and post-2015 activations — pure engineering

**GELU (2016)**, **Swish (2017)**, **Mish (2019)** — these have no meaningful biological basis.
They were designed by people doing exactly what you might expect: understanding what mathematical
properties the activation function needs to have for a very deep network to train well, and then
finding a smooth function that satisfies those properties.

The reasoning:

- *Dying ReLU*: a neuron with ReLU can get stuck permanently outputting zero if its weights push
  its input permanently negative — gradient is zero, no recovery possible. GELU never hard-zeros;
  it tapers smoothly, so recovery is always possible.
- *Smoothness*: in very deep networks (96 layers in GPT-3), the tiny accumulated numerical
  advantage of a smooth activation over a kinked one compounds into measurable differences in
  final loss.

No one checked a brain scan. GELU and its contemporaries are products of understanding the
**mathematics of gradient flow through compositions of functions**, not neuroscience experiments.

---

## Convolution — directly from visual cortex

**Conv2D** is perhaps the most directly biologically motivated modern layer type. Hubel and
Wiesel's 1959–62 experiments did not just identify simple cells (the ReLU connection above) —
they also described **receptive fields**: each cell responds only to a small local region of
the visual field. Neurons in layers further from the retina have progressively larger receptive
fields, combining the responses of earlier local detectors into more complex, position-invariant
features. Yann LeCun's **LeNet (1989)** translated this directly: convolutional filters with
local receptive fields, weight sharing (the filter is the same kernel applied across the whole
image — just like a cortical neuron that detects a specific pattern anywhere in its region).

---

## Recurrent networks — working memory and sequence

**RNNs** (1980s) were inspired by the observation that brain processing is inherently recurrent —
signals flow in loops, not just forward. The concept of a **hidden state** that persists across
time steps is a rough analogy to short-term working memory. **LSTM (1997, Hochreiter & Schmidhuber)**
was technically motivated (solving vanishing gradients in RNNs through gating mechanisms) but
the gate metaphor — input gate, forget gate, output gate — has a loose analogy to how attention
and working memory interact in the prefrontal cortex.

---

## Attention — the most significant biology-meets-engineering convergence

**Attentional modulation** in neuroscience — how top-down signals from prefrontal cortex suppress
irrelevant sensory inputs and amplify relevant ones — was studied experimentally for decades before
Transformers. Neuroscientists documented this in the 1980s–90s: a neuron's response to its
preferred stimulus can be doubled or halved depending on whether the subject is "attending" to it.

The researchers who coined "attention mechanism" in neural translation (Bahdanau et al., 2014;
Vaswani et al., 2017 — "Attention Is All You Need") were aware of the neuroscience concept.
However, the mathematical form they chose — query-key-value dot products with softmax weighting —
is their own engineering construction. The *idea* has biological motivation; the *mechanism* is
mathematical.

**Multi-head attention** has a further parallel: different heads learn to attend to different
linguistic or structural aspects simultaneously (one head tracks syntax, another coreference,
another positional patterns). This mirrors how different cortical areas process the same sensory
input in parallel, each specialised for a different aspect.

---

## What the brain does that our models still don't

| Brain mechanism | Status in deep learning |
|---|---|
| Spike timing (not just rate) | Not modelled — rate coding is the standard approximation |
| Neuromodulation (dopamine rescaling learning) | Partially: learning rate schedulers are a weak analogy; not mechanistic |
| Massive recurrence (most brain connections are feedback, not feedforward) | Mostly absent — Transformers have residual connections but no true recurrence |
| Hebbian / STDP local learning | Not used — backpropagation sends a global error signal, not a local causal one |
| Energy efficiency (~20W for the whole brain) | State-of-the-art models use kilowatts; neuromorphic hardware is an active research area |

Backpropagation is widely agreed **not** to be how biological learning actually works — neurons
do not send error signals backwards along axons. It was chosen because it is mathematically sound
and practically tractable, not because it models biology.

---

## Reference table — NNSpire layers and their historical origin

| NNSpire layer / activation | Biological basis | Historical origin | Era |
|---|---|---|---|
| `Dense` (fully-connected) | None | Mathematical generalization of the perceptron | 1958 (Rosenblatt) |
| Step / threshold | ✅ Direct — all-or-nothing action potential | McCulloch-Pitts unit | 1943 |
| `Sigmoid` | ✅ Approximate — S-shaped f-I firing rate curve | Rumelhart, Hinton, Williams (1986 backprop paper) | 1986 |
| `Tanh` | ✅ Approximate — variant of sigmoid firing rate curve | Same era as Sigmoid, zero-centred variant | 1980s |
| `ReLU` | ⚠️ Partial — Hubel & Wiesel half-wave rectification (1959) | Reintroduced for gradient reasons (Glorot 2011, AlexNet 2012) | 2011–2012 |
| `GELU` | ❌ None | Pure engineering — smooth dying-neuron fix (Hendrycks & Gimpel, 2016) | 2016 |
| Swish / Mish | ❌ None | Pure engineering — smooth activation search | 2017–2019 |
| `Softmax` | ❌ None | Mathematical (multi-class probability normalisation) | 1959 (Luce's choice axiom) |
| `Conv2D` | ✅ Direct — Hubel & Wiesel local receptive fields, simple cells | LeCun LeNet (1989) | 1989 |
| `MaxPool` | ⚠️ Partial — complex cells (position-invariant response) | LeCun (1989) | 1989 |
| `Embedding` | ❌ None | Mathematical (learnable lookup / distributed representation) | Word2Vec (2013) |
| Positional encoding | ❌ None | Engineering — compensates for attention's set-theoretic blindness | Vaswani et al. (2017) |
| `MultiHeadAttention` | ⚠️ Concept — prefrontal attentional modulation (1980s neuroscience) | Mathematical mechanism is original (Bahdanau 2014, Vaswani 2017) | 2017 |
| `LayerNorm` | ❌ None | Engineering — stabilize activation magnitudes in deep nets | Ba et al. (2016) |
| RNN hidden state | ⚠️ Loose — working memory / short-term potentiation | Jordan (1986), Elman (1990) | 1986–1990 |
| LSTM gates | ⚠️ Loose — working memory gating (prefrontal / hippocampus) | Hochreiter & Schmidhuber (1997) — technically motivated | 1997 |

> ✅ = direct or strong biological motivation  
> ⚠️ = partial or analogical motivation  
> ❌ = no meaningful biological basis — pure mathematics or engineering

---

## Further reading

- McCulloch & Pitts (1943) — *A logical calculus of the ideas immanent in nervous activity*
- Hubel & Wiesel (1959, 1962) — receptive fields in the cat and macaque visual cortex
- Rumelhart, Hinton & Williams (1986) — *Learning representations by back-propagating errors*
- LeCun et al. (1989, 1998) — LeNet / convolutional networks applied to digit recognition
- Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory*
- Glorot & Bengio (2011) — *Understanding the difficulty of training deep feedforward networks*
- Krizhevsky, Sutskever & Hinton (2012) — AlexNet / ImageNet classification with deep CNNs
- Bahdanau et al. (2014) — *Neural Machine Translation by Jointly Learning to Align and Translate*
- Vaswani et al. (2017) — *Attention Is All You Need*
- Hendrycks & Gimpel (2016) — *Gaussian Error Linear Units (GELUs)*
