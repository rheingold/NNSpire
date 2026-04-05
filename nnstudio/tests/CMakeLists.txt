# NNStudio unit and integration tests
# Phase 1+: each phase adds a test subdirectory

include(GoogleTest)

# ─── Core engine tests (Phase 1) ─────────────────────────────────────────────
add_executable(test-core
    core/test_tensor.cpp
    core/test_activations.cpp
    core/test_trainer_xor.cpp
    # core/test_dense_layer.cpp
    # core/test_losses.cpp
    # core/test_optimizers.cpp
    # core/test_compute_graph.cpp
    # core/test_onnx_roundtrip.cpp
)
target_link_libraries(test-core PRIVATE
    nnstudio-builtin nnstudio-backend-cpu GTest::gtest_main)
# Include paths are now provided by nnstudio-builtin PUBLIC includes
gtest_discover_tests(test-core)

# ─── Plugin SDK tests (Phase 2) ──────────────────────────────────────────────
# Phase 2/4/5 tests are commented out — uncomment as each phase is implemented
# add_executable(test-plugin-sdk
#     plugin/test_trust_verifier.cpp
#     plugin/test_plugin_loader.cpp
# )
# target_link_libraries(test-plugin-sdk PRIVATE nnstudio-core nnstudio-trust GTest::gtest_main)
# gtest_discover_tests(test-plugin-sdk)

# add_executable(test-pipeline
#     pipeline/test_text_adapter.cpp
# )
# target_link_libraries(test-pipeline PRIVATE nnstudio-core nnstudio-pipeline GTest::gtest_main)
# gtest_discover_tests(test-pipeline)

# add_executable(test-deployment
#     deployment/test_onnx_export.cpp
# )
# target_link_libraries(test-deployment PRIVATE nnstudio-core nnstudio-deployment GTest::gtest_main)
# gtest_discover_tests(test-deployment)
