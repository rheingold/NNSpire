# NNStudio unit and integration tests
# Layout mirrors include/ and src/:  core/  builtin/  plugin-api/  pipeline/  deployment/

include(GoogleTest)

# ─── Core engine tests ────────────────────────────────────────────────────────
# SUT: Tensor, ComputeGraph, EarlyStopping, Trainer (integration XOR)
add_executable(test-core
    core/test_tensor.cpp
    core/test_compute_graph.cpp
    core/test_early_stopping.cpp
    core/test_trainer_xor.cpp
    core/test_checkpoint.cpp
    core/test_compat.cpp
    # core/test_onnx_roundtrip.cpp  # deferred
)
target_link_libraries(test-core PRIVATE
    nnstudio-builtin nnstudio-backend-cpu GTest::gtest_main)
gtest_discover_tests(test-core)

# ─── Built-in layer / loss / optimizer tests ─────────────────────────────────
# SUT: Conv2D, Embedding, Activations, NormLayers, Losses, Optimizers
add_executable(test-builtin
    builtin/test_activations.cpp
    builtin/test_layers.cpp
    builtin/test_losses.cpp
    builtin/test_optimizers.cpp
    builtin/test_conv2d.cpp
    builtin/test_embedding.cpp
    # builtin/test_multihead_attention.cpp  # Phase 1 remaining
)
target_link_libraries(test-builtin PRIVATE
    nnstudio-builtin nnstudio-backend-cpu GTest::gtest_main)
gtest_discover_tests(test-builtin)

# ─── Plugin-API tests (requires NN_ENABLE_TRUST=ON) ──────────────────────────
# SUT: TrustStore, TrustVerifier, TrustUpdateHandler, PluginLoader
if(NN_ENABLE_TRUST)
    add_executable(test-plugin-api
        plugin-api/test_trust_store.cpp
        plugin-api/test_trust_verifier.cpp
        plugin-api/test_plugin_loader.cpp
        # plugin-api/test_trust_update_handler.cpp  # Phase 2 — needs .tup fixture
    )
    target_link_libraries(test-plugin-api PRIVATE
        nnstudio-trust GTest::gtest_main)
    gtest_discover_tests(test-plugin-api)
endif()

# ─── Built-in plugin tests (no dlopen — compile plugin source directly) ───────
# Phase 2 milestone: PluginLoader can successfully load BPE tokenizer.
# Here we test the C ABI contract directly to keep tests fast and portable.
add_executable(test-plugins
    plugin-api/test_bpe_tokenizer.cpp
    ${CMAKE_SOURCE_DIR}/plugins/bpe_tokenizer/BpeTokenizerPlugin.cpp
)
target_include_directories(test-plugins PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)
target_link_libraries(test-plugins PRIVATE
    nnstudio-plugin-api GTest::gtest_main)
target_compile_options(test-plugins PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti -fno-exceptions>
    $<$<CXX_COMPILER_ID:MSVC>:/GR- /EHs->
)
gtest_discover_tests(test-plugins)

# ─── Pipeline tests (Phase 3) ────────────────────────────────────────────────
# add_executable(test-pipeline
#     pipeline/test_text_adapter.cpp
# )
# target_link_libraries(test-pipeline PRIVATE nnstudio-pipeline GTest::gtest_main)
# gtest_discover_tests(test-pipeline)

# ─── Deployment tests (Phase 4) ──────────────────────────────────────────────
# add_executable(test-deployment
#     deployment/test_onnx_export.cpp
# )
# target_link_libraries(test-deployment PRIVATE nnstudio-deployment GTest::gtest_main)

# ─── Cross-framework compatibility layer tests ────────────────────────────────
# Requires torch_compat.h — verifies the torch:: alias surface compiles and runs.
add_executable(test-compat
    compat/test_torch_compat.cpp
)
target_link_libraries(test-compat PRIVATE
    nnstudio-builtin nnstudio-backend-cpu GTest::gtest_main)
gtest_discover_tests(test-compat)
# gtest_discover_tests(test-deployment)
