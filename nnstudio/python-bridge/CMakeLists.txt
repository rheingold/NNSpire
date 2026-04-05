# Python pybind11 bridge — LGPL v3
# Phase 2
# Exposes nnstudio-core to Python as the `nnstudio` package.
# Mirror: every public C++ API in nnstudio-core has an identical Python binding here.

pybind11_add_module(nnstudio_pyd
    # bindings/module.cpp         ← pybind11 PYBIND11_MODULE definition
    # bindings/tensor.cpp         ← Tensor<float> bindings
    # bindings/layers.cpp         ← all Layer subclass bindings
    # bindings/activations.cpp
    # bindings/losses.cpp
    # bindings/optimizers.cpp
    # bindings/graph.cpp          ← ComputeGraph
    # bindings/training.cpp       ← Trainer, callbacks
    # bindings/formats.cpp        ← OnnxIO, NNSFormat
    # bindings/backends.cpp       ← BackendRegistry
    # bindings/pipeline.cpp       ← pipeline stage bindings (Phase 4)
    # bindings/deployment.cpp     ← runner client bindings (Phase 5)
)

target_link_libraries(nnstudio_pyd PRIVATE
    nnstudio-core
    nnstudio-pipeline
    nnstudio-deployment
    nnstudio-trust
)

set_target_properties(nnstudio_pyd PROPERTIES
    OUTPUT_NAME "nnstudio"     # import nnstudio
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/nnstudio"
)

# Copy pure-Python package files (runners/, plugins/, pyproject.toml)
# alongside the compiled .pyd/.so
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/nnstudio/__init__.py"
    "${CMAKE_BINARY_DIR}/python/nnstudio/__init__.py"
    COPYONLY
)
