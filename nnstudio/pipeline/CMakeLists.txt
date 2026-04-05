# NNStudio pipeline — input adapters, tokenization chain, context, execution, output
# Phase 4 — INTERFACE stub until NN_ENABLE_APP is ON

add_library(nnstudio-pipeline INTERFACE)
target_include_directories(nnstudio-pipeline INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_link_libraries(nnstudio-pipeline INTERFACE nnstudio-core)

# ─── Full pipeline (Phase 4, requires Qt) ────────────────────────────────────
# Uncomment when NN_ENABLE_APP=ON and stage sources are written:
# add_library(nnstudio-pipeline-full STATIC
#     input/TextInputAdapter.cpp
#     ...
# )
# target_link_libraries(nnstudio-pipeline-full PUBLIC nnstudio-core nnstudio-plugin-api
#                                              PRIVATE Qt::Network)
