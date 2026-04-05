# Built-in reference plugins (MIT licensed)
# Phase 2: BPE tokenizer, FAISS index, example custom activation
# Each plugin is both a C++ shared library AND a Python package

# ─── BPE tokenizer plugin ────────────────────────────────────────────────────
# @kb: ai-standards-kb/annexes/A06-Tokenization-Deep-Dive.md
add_library(plugin-bpe-tokenizer SHARED
    # bpe_tokenizer/BpeTokenizerPlugin.cpp
)
target_link_libraries(plugin-bpe-tokenizer PRIVATE nnstudio-plugin-api nnstudio-core)
set_target_properties(plugin-bpe-tokenizer PROPERTIES
    PREFIX ""   # no lib prefix on Linux — matches nnstudio_plugin.h conventions
)

# ─── FAISS context source plugin ─────────────────────────────────────────────
# @kb: ai-standards-kb/annexes/A02-Vector-Databases.md
add_library(plugin-faiss-context SHARED
    # faiss_context/FaissContextPlugin.cpp
)
target_link_libraries(plugin-faiss-context PRIVATE nnstudio-plugin-api nnstudio-core)
set_target_properties(plugin-faiss-context PROPERTIES PREFIX "")

# ─── Example custom activation plugin ────────────────────────────────────────
add_library(plugin-example-activation SHARED
    # example_activation/ExampleActivationPlugin.cpp
)
target_link_libraries(plugin-example-activation PRIVATE nnstudio-plugin-api nnstudio-core)
set_target_properties(plugin-example-activation PROPERTIES PREFIX "")

# Python-side of each built-in plugin lives in python-bridge/nnstudio/plugins/
