# NNSpire pipeline — input adapters, tokenization chain, context, execution, output
# Phase 4 — INTERFACE stub until NN_ENABLE_APP is ON

add_library(NNSpire-pipeline INTERFACE)
target_include_directories(NNSpire-pipeline INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_link_libraries(NNSpire-pipeline INTERFACE NNSpire-core)

# ─── Full pipeline (Phase 4, requires Qt) ────────────────────────────────────
# Uncomment when NN_ENABLE_APP=ON and stage sources are written:
# add_library(NNSpire-pipeline-full STATIC
#     input/TextInputAdapter.cpp
#     ...
# )
# target_link_libraries(NNSpire-pipeline-full PUBLIC NNSpire-core NNSpire-plugin-api
#                                              PRIVATE Qt::Network)
