# NNStudio deployment — export wizards, runner connectors, bundle builder
# Phase 5 — INTERFACE stub until NN_ENABLE_APP is ON
# @kb: ai-standards-kb/standards/02-ONNX.md

add_library(nnstudio-deployment INTERFACE)
target_include_directories(nnstudio-deployment INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_link_libraries(nnstudio-deployment INTERFACE nnstudio-core)

# ─── Full deployment (Phase 5, requires Qt) ───────────────────────────────────
# Uncomment when NN_ENABLE_APP=ON and export sources are written:
# add_library(nnstudio-deployment-full STATIC
#     export/OnnxExportWizard.cpp
#     ...
# )
# target_link_libraries(nnstudio-deployment-full PUBLIC nnstudio-core
#                                                PRIVATE Qt::Network OpenSSL::SSL)
