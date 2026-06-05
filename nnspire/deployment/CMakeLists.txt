# NNSpire deployment — export wizards, runner connectors, bundle builder
# Phase 5 — INTERFACE stub until NN_ENABLE_APP is ON
# @kb: ai-standards-kb/standards/02-ONNX.md

add_library(NNSpire-deployment INTERFACE)
target_include_directories(NNSpire-deployment INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_link_libraries(NNSpire-deployment INTERFACE NNSpire-core)

# ─── Full deployment (Phase 5, requires Qt) ───────────────────────────────────
# Uncomment when NN_ENABLE_APP=ON and export sources are written:
# add_library(NNSpire-deployment-full STATIC
#     export/OnnxExportWizard.cpp
#     ...
# )
# target_link_libraries(NNSpire-deployment-full PUBLIC NNSpire-core
#                                                PRIVATE Qt::Network OpenSSL::SSL)
