/* ============================================================================
 * module.cpp — pybind11 Python bindings for NNStudio
 * LGPL v3
 *
 * Exposes nnstudio-core, nnstudio-builtin to Python as the `nnstudio` package.
 *
 * NAMING CONVENTION
 * ─────────────────
 * All public names mirror PyTorch / LibTorch for maximum drop-in compatibility:
 *   nnstudio.nn.Linear(in, out)       ← matches torch.nn.Linear
 *   nnstudio.optim.Adam(lr=1e-3)      ← matches torch.optim.Adam
 *   nnstudio.nn.functional.relu(x)    ← matches torch.nn.functional.relu
 *
 * This is a DESIGN CONSTRAINT, not a nice-to-have.
 * See docs/blueprints.md §9.
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
 * ============================================================================ */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <core/Tensor.h>
#include <core/DType.h>
#include <core/Device.h>
#include <core/Layer.h>
#include <builtin/layers/Dense.h>
#include <builtin/layers/Conv2D.h>
#include <builtin/layers/Embedding.h>
#include <builtin/layers/MultiHeadAttention.h>
#include <builtin/layers/NormLayers.h>
#include <builtin/activations/Activations.h>
#include <builtin/activations/Functors.h>
#include <builtin/activations/FnLayer.h>
#include <builtin/losses/Losses.h>
#include <builtin/optimizers/Optimizers.h>
#include <core/ComputeGraph.h>

namespace py = pybind11;
using namespace pybind11::literals;

// short aliases for cleaner code below
using Tensor    = nnstudio::core::Tensor;
using Shape     = nnstudio::core::Shape;
using DType     = nnstudio::core::DType;
using Device    = nnstudio::core::Device;
using ILayer    = nnstudio::core::ILayer;
using Parameter = nnstudio::core::Parameter;

namespace nsl = nnstudio::builtin::layers;
namespace nsa = nnstudio::builtin::activations;
namespace nso = nnstudio::builtin::optimizers;
namespace nsloss = nnstudio::builtin::losses;

// ─── Module entry point ──────────────────────────────────────────────────────
PYBIND11_MODULE(nnstudio, m) {
    m.doc() = "NNStudio — Neural Network engine Python bindings (torch-compatible API)";

    // ── DType enum ───────────────────────────────────────────────────────────
    py::enum_<DType>(m, "DType")
        .value("float32", DType::Float32)
        .value("float16", DType::Float16)
        .value("int8",    DType::Int8)
        .value("int32",   DType::Int32)
        .value("bool_",   DType::Bool)
        .export_values();

    // ── Device enum ──────────────────────────────────────────────────────────
    py::enum_<Device>(m, "Device")
        .value("cpu",  Device::CPU)
        .value("cuda", Device::CUDA)
        .export_values();

    // ── Tensor ───────────────────────────────────────────────────────────────
    // Public surface mirrors torch.Tensor where meaningful for a CPU engine.
    py::class_<Tensor>(m, "Tensor")
        .def(py::init([](std::vector<int64_t> shape, DType dtype, Device dev) {
                return Tensor(shape, dtype, dev);
             }),
             "shape"_a, "dtype"_a = DType::Float32, "device"_a = Device::CPU)
        // Properties
        .def_property_readonly("shape",  [](const Tensor& t) { return t.shape(); })
        .def_property_readonly("dtype",  &Tensor::dtype)
        .def_property_readonly("device", &Tensor::device)
        .def_property_readonly("ndim",   &Tensor::ndim)
        // torch.Tensor.numel()
        .def("numel",    &Tensor::numel)
        .def("itemsize", &Tensor::itemsize)
        // Scalar extraction — mirrors torch.Tensor.item()
        .def("item", [](const Tensor& t) -> py::object {
            if (t.dtype() == DType::Float32) return py::cast(t.item<float>());
            if (t.dtype() == DType::Int32)   return py::cast(t.item<int32_t>());
            if (t.dtype() == DType::Int8)    return py::cast(t.item<int8_t>());
            throw std::runtime_error("item(): unsupported dtype");
        })
        // numpy() — CPU copy as a float32 numpy array (Phase 1: Float32 only)
        .def("numpy", [](const Tensor& t) -> py::array_t<float> {
            if (t.dtype() != DType::Float32)
                throw std::runtime_error("numpy(): only Float32 tensors supported in Phase 1");
            return py::array_t<float>(
                t.shape(),       // shape
                t.data()         // pointer (data is copied into numpy object)
            );
        })
        // requires_grad
        .def_property("requires_grad",
            &Tensor::requiresGrad, &Tensor::setRequiresGrad)
        .def("clone", &Tensor::clone)
        .def("__repr__", [](const Tensor& t) {
            std::string s = "Tensor(shape=[";
            for (size_t i = 0; i < t.shape().size(); ++i) {
                if (i) s += ", ";
                s += std::to_string(t.shape()[i]);
            }
            s += "])";
            return s;
        });

    // ── Free tensor factory functions — torch.zeros / ones / full / rand ─────
    m.def("zeros", [](std::vector<int64_t> shape, DType dtype, Device dev) {
        return Tensor::zeros(shape, dtype, dev);
    }, "shape"_a, "dtype"_a = DType::Float32, "device"_a = Device::CPU);

    m.def("ones", [](std::vector<int64_t> shape, DType dtype, Device dev) {
        return Tensor::ones(shape, dtype, dev);
    }, "shape"_a, "dtype"_a = DType::Float32, "device"_a = Device::CPU);

    m.def("full", [](std::vector<int64_t> shape, float value, DType dtype, Device dev) {
        return Tensor::full(shape, value, dtype, dev);
    }, "shape"_a, "fill_value"_a, "dtype"_a = DType::Float32, "device"_a = Device::CPU);

    // ── nn submodule ─────────────────────────────────────────────────────────
    auto nn = m.def_submodule("nn", "nnstudio.nn — torch-compatible layer API");

    // ── ILayer base (exposed as nn.Module) ───────────────────────────────────
    py::class_<ILayer>(nn, "Module")
        .def("typeName",  [](const ILayer& l) { return std::string(l.typeName()); })
        .def("name",      [](const ILayer& l) { return l.name(); })
        .def("isBuilt",   &ILayer::isBuilt)
        .def("train",     [](ILayer& l) { l.train(); })
        .def("eval",      [](ILayer& l) { l.eval(); })
        .def("training",  &ILayer::training)
        .def("zeroGrad",  &ILayer::zeroGrad)
        .def("forward", [](ILayer& l, const Tensor& x) {
            auto r = l.forward(x);
            if (!r.ok()) throw std::runtime_error(r.error().message);
            return r.value();
        })
        .def("build", [](ILayer& l, std::vector<int64_t> shape) {
            auto r = l.build(shape);
            if (!r.ok()) throw std::runtime_error(r.error().message);
            return r.value();
        })
        .def("backward", [](ILayer& l, const Tensor& grad) {
            auto r = l.backward(grad);
            if (!r.ok()) throw std::runtime_error(r.error().message);
            return r.value();
        })
        .def("parameters", [](ILayer& l) { return l.parameters(); },
             py::return_value_policy::reference_internal)
        .def("setName", [](ILayer& l, const std::string& n) { l.setName(n); });

    // ── Parameter ─────────────────────────────────────────────────────────────
    // A trainable weight/bias tensor and its gradient accumulator.
    py::class_<Parameter>(nn, "Parameter")
        .def_readwrite("name",   &Parameter::name)
        .def_readwrite("frozen", &Parameter::frozen)
        .def_property("data",
            [](Parameter& p) -> Tensor& { return p.tensor; },
            [](Parameter& p, const Tensor& t) { p.tensor = t; })
        .def_property_readonly("grad", [](const Parameter& p) -> py::object {
            if (!p.tensor.hasGrad()) return py::none();
            return py::cast(p.tensor.grad());
        })
        .def("__repr__", [](const Parameter& p) {
            return "Parameter(name='" + p.name + "', numel=" +
                   std::to_string(p.tensor.numel()) + ")";
        });

    // ── Linear (Dense) ───────────────────────────────────────────────────────
    // torch.nn.Linear(in_features, out_features, bias=True)
    // NNStudio Dense(outFeatures, useBias, init); in_features is informational only.
    py::class_<nsl::Dense, ILayer>(nn, "Linear")
        .def(py::init([](int64_t /*in*/, int64_t out, bool bias) {
                return nsl::Dense(out, bias);
             }),
             "in_features"_a, "out_features"_a, "bias"_a = true,
             R"doc(Fully-connected linear transformation. y = x @ W^T + b.
Matches torch.nn.Linear(in_features, out_features, bias=True).)doc");

    // ── Conv2d ────────────────────────────────────────────────────────────────
    // torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
    py::class_<nsl::Conv2D, ILayer>(nn, "Conv2d")
        .def(py::init([](int64_t /*in_ch*/, int64_t out_ch,
                         int64_t kernel, int64_t stride, int64_t padding, bool bias) {
                return nsl::Conv2D(out_ch, kernel, stride, padding, bias);
             }),
             "in_channels"_a, "out_channels"_a,
             "kernel_size"_a = 3, "stride"_a = 1, "padding"_a = 0, "bias"_a = true);

    // ── Embedding ─────────────────────────────────────────────────────────────
    // torch.nn.Embedding(num_embeddings, embedding_dim)
    py::class_<nsl::Embedding, ILayer>(nn, "Embedding")
        .def(py::init<int64_t, int64_t>(),
             "num_embeddings"_a, "embedding_dim"_a);

    // ── MultiheadAttention ────────────────────────────────────────────────────
    // torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True)
    // NNStudio: MultiHeadAttention(numHeads, embDim, causal, dropout) — reversed!
    py::class_<nsl::MultiHeadAttention, ILayer>(nn, "MultiheadAttention")
        .def(py::init([](int64_t embed_dim, int64_t num_heads,
                         float dropout, bool /*bias*/) {
                return nsl::MultiHeadAttention(num_heads, embed_dim,
                                               /*causal=*/false, dropout);
             }),
             "embed_dim"_a, "num_heads"_a, "dropout"_a = 0.0f, "bias"_a = true);

    // ── BatchNorm1d ───────────────────────────────────────────────────────────
    // torch.nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True)
    // NNStudio: BatchNorm1d(eps, momentum); num_features and affine not stored.
    py::class_<nsl::BatchNorm1d, ILayer>(nn, "BatchNorm1d")
        .def(py::init([](int64_t /*num_features*/, float eps, float momentum,
                         bool /*affine*/) {
                return nsl::BatchNorm1d(eps, momentum);
             }),
             "num_features"_a, "eps"_a = 1e-5f, "momentum"_a = 0.1f, "affine"_a = true);

    // ── LayerNorm ─────────────────────────────────────────────────────────────
    // torch.nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)
    py::class_<nsl::LayerNorm, ILayer>(nn, "LayerNorm")
        .def(py::init([](int64_t normalized_shape, float eps,
                         bool /*elementwise_affine*/) {
                return nsl::LayerNorm(normalized_shape, eps);
             }),
             "normalized_shape"_a, "eps"_a = 1e-5f, "elementwise_affine"_a = true);

    // ── Dropout ───────────────────────────────────────────────────────────────
    py::class_<nsl::Dropout, ILayer>(nn, "Dropout")
        .def(py::init<float>(), "p"_a = 0.5f);

    // ── Activations ───────────────────────────────────────────────────────────
    py::class_<nsa::ReLU, ILayer>(nn, "ReLU")
        .def(py::init<>());

    py::class_<nsa::Sigmoid, ILayer>(nn, "Sigmoid")
        .def(py::init<>());

    py::class_<nsa::TanhAct, ILayer>(nn, "Tanh")
        .def(py::init<>());

    py::class_<nsa::GELU, ILayer>(nn, "GELU")
        .def(py::init<>());

    py::class_<nsa::Softmax, ILayer>(nn, "Softmax")
        .def(py::init<>());                      // dim handled inside forward

    py::class_<nsa::LeakyReLU, ILayer>(nn, "LeakyReLU")
        .def(py::init<float>(), "negative_slope"_a = 0.01f);

    // ── Sequential ────────────────────────────────────────────────────────────
    // torch.nn.Sequential(*layers)
    // Python usage: model = nnstudio.nn.Sequential(); model.add(layer)
    py::class_<nnstudio::core::Sequential, ILayer>(nn, "Sequential")
        .def(py::init<>())
        .def("add", [](nnstudio::core::Sequential& s,
                       std::shared_ptr<ILayer> l) -> nnstudio::core::Sequential& {
            return s.add(l);
        }, "layer"_a, py::return_value_policy::reference_internal)
        .def("size", &nnstudio::core::Sequential::size)
        .def("layers", [](const nnstudio::core::Sequential& s) {
            return s.layers();
        })
        .def("build", [](nnstudio::core::Sequential& s, std::vector<int64_t> shape) {
            auto r = s.build(shape);
            if (!r.ok()) throw std::runtime_error(r.error().message);
            return r.value();
        })
        .def("forward", [](nnstudio::core::Sequential& s, const Tensor& x) {
            auto r = s.forward(x);
            if (!r.ok()) throw std::runtime_error(r.error().message);
            return r.value();
        });

    // ── Loss functions ────────────────────────────────────────────────────────
    // ILoss is NOT an ILayer; it has a different forward(pred, target) signature.
    py::class_<nsloss::MSE>(nn, "MSELoss")
        .def(py::init<>())
        .def("forward", [](nsloss::MSE& self, const Tensor& pred, const Tensor& tgt) {
            auto r = self.compute(pred, tgt);
            if (!r.ok()) throw std::runtime_error(r.error().message);
            return r.value();
        })
        .def("gradient", [](nsloss::MSE& self, const Tensor& pred, const Tensor& tgt) {
            auto r = self.gradient(pred, tgt);
            if (!r.ok()) throw std::runtime_error(r.error().message);
            return r.value();
        });

    py::class_<nsloss::CrossEntropy>(nn, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward", [](nsloss::CrossEntropy& self, const Tensor& pred, const Tensor& tgt) {
            auto r = self.compute(pred, tgt);
            if (!r.ok()) throw std::runtime_error(r.error().message);
            return r.value();
        })
        .def("gradient", [](nsloss::CrossEntropy& self, const Tensor& pred, const Tensor& tgt) {
            auto r = self.gradient(pred, tgt);
            if (!r.ok()) throw std::runtime_error(r.error().message);
            return r.value();
        });

    py::class_<nsloss::BCE>(nn, "BCELoss")
        .def(py::init<>())
        .def("forward", [](nsloss::BCE& self, const Tensor& pred, const Tensor& tgt) {
            auto r = self.compute(pred, tgt);
            if (!r.ok()) throw std::runtime_error(r.error().message);
            return r.value();
        })
        .def("gradient", [](nsloss::BCE& self, const Tensor& pred, const Tensor& tgt) {
            auto r = self.gradient(pred, tgt);
            if (!r.ok()) throw std::runtime_error(r.error().message);
            return r.value();
        });

    // BCEWithLogitsLoss — Phase 1: delegates to BCE (no sigmoid pre-processing)
    nn.attr("BCEWithLogitsLoss") = nn.attr("BCELoss");

    py::class_<nsloss::HuberLoss>(nn, "HuberLoss")
        .def(py::init<float>(), "delta"_a = 1.0f)
        .def("forward", [](nsloss::HuberLoss& self, const Tensor& pred, const Tensor& tgt) {
            auto r = self.compute(pred, tgt);
            if (!r.ok()) throw std::runtime_error(r.error().message);
            return r.value();
        })
        .def("gradient", [](nsloss::HuberLoss& self, const Tensor& pred, const Tensor& tgt) {
            auto r = self.gradient(pred, tgt);
            if (!r.ok()) throw std::runtime_error(r.error().message);
            return r.value();
        });

    // ── nn.functional submodule ───────────────────────────────────────────────
    auto fn = nn.def_submodule("functional",
                               "nnstudio.nn.functional — stateless activation ops");

    fn.def("relu", [](const Tensor& x) {
        return nsa::ReLUFn{}.forward(x).output;
    }, "input"_a);

    fn.def("sigmoid", [](const Tensor& x) {
        return nsa::SigmoidFn{}.forward(x).output;
    }, "input"_a);

    fn.def("tanh", [](const Tensor& x) {
        return nsa::TanhActFn{}.forward(x).output;
    }, "input"_a);

    fn.def("gelu", [](const Tensor& x) {
        return nsa::GELUFn{}.forward(x).output;
    }, "input"_a);

    fn.def("softmax", [](const Tensor& x, int64_t /*dim*/) {
        return nsa::SoftmaxFn{}.forward(x).output;
    }, "input"_a, "dim"_a = -1);

    fn.def("leaky_relu", [](const Tensor& x, float negative_slope) {
        return nsa::LeakyReLUFn{negative_slope}.forward(x).output;
    }, "input"_a, "negative_slope"_a = 0.01f);

    fn.def("dropout", [](const Tensor& x, float p, bool training) {
        nsl::Dropout d(p);
        d.setTraining(training);
        d.build(x.shape());
        auto r = d.forward(x);
        if (!r.ok()) throw std::runtime_error(r.error().message);
        return r.value();
    }, "input"_a, "p"_a = 0.5f, "training"_a = true);

    // ── ComputeGraph ──────────────────────────────────────────────────────────
    // nn.ComputeGraph — DAG-based execution engine, exposes the same ILayer API.
    using CG = nnstudio::core::ComputeGraph;
    py::class_<CG, ILayer>(nn, "ComputeGraph")
        .def(py::init<>())
        // Graph construction
        .def("input",  &CG::input,
             "Allocate the input node (call once). Returns NodeId 0.")
        .def("record", [](CG& g, ILayer& l, nnstudio::core::NodeId id) {
            return g.record(&l, id);
        }, py::keep_alive<1, 2>(),
           "Record a layer op: inputId → layer → new NodeId.")
        .def("setOutput",    &CG::setOutput)
        .def("outputNodeId", &CG::outputNodeId)
        // Trace mode
        .def("setTraceMode", &CG::setTraceMode)
        .def("traceMode",    &CG::traceMode)
        // JSON serialization (for graph visualization)
        .def("toJson", &CG::toJson)
        .def_static("registeredLayerTypes", &CG::registeredLayerTypes)
        // ILayer interface overrides already inherited from nn.Module
        .def("parameters", [](CG& g) { return g.parameters(); },
             py::return_value_policy::reference_internal);

    // ── optim submodule ───────────────────────────────────────────────────────
    auto optim = m.def_submodule("optim",
                                 "nnstudio.optim — torch-compatible optimizer API");

    // Helper lambda: collect params from a list of layers
    // Usage: optimizer.step(model.parameters())  — not yet; for now step(params_vec)

    py::class_<nso::SGD>(optim, "SGD")
        .def(py::init<float, float, float>(),
             "lr"_a = 1e-2f, "momentum"_a = 0.0f, "weight_decay"_a = 0.0f)
        .def("step", [](nso::SGD& o, std::vector<Parameter*> params) {
            o.step(params);
        })
        .def("zero_grad", [](nso::SGD& /*o*/, std::vector<Parameter*> params) {
            for (auto* p : params) p->tensor.zeroGrad();
        });

    py::class_<nso::Adam>(optim, "Adam")
        .def(py::init<float, float, float, float, float>(),
             "lr"_a = 1e-3f, "beta1"_a = 0.9f, "beta2"_a = 0.999f,
             "eps"_a = 1e-8f, "weight_decay"_a = 0.0f)
        .def("step", [](nso::Adam& o, std::vector<Parameter*> params) {
            o.step(params);
        })
        .def("zero_grad", [](nso::Adam& /*o*/, std::vector<Parameter*> params) {
            for (auto* p : params) p->tensor.zeroGrad();
        });

    py::class_<nso::AdamW, nso::Adam>(optim, "AdamW")
        .def(py::init<float, float, float, float, float>(),
             "lr"_a = 1e-3f, "beta1"_a = 0.9f, "beta2"_a = 0.999f,
             "eps"_a = 1e-8f, "weight_decay"_a = 1e-2f)
        .def("step", [](nso::AdamW& o, std::vector<Parameter*> params) {
            o.step(params);
        });

    py::class_<nso::RMSProp>(optim, "RMSprop")
        .def(py::init<float, float, float, float>(),
             "lr"_a = 1e-2f, "alpha"_a = 0.99f, "eps"_a = 1e-8f,
             "weight_decay"_a = 0.0f)
        .def("step", [](nso::RMSProp& o, std::vector<Parameter*> params) {
            o.step(params);
        })
        .def("zero_grad", [](nso::RMSProp& /*o*/, std::vector<Parameter*> params) {
            for (auto* p : params) p->tensor.zeroGrad();
        });

    // ── Version ───────────────────────────────────────────────────────────────
    m.attr("__version__") = "0.1.0";
}
