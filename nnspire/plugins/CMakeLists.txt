# Built-in reference plugins (MIT licensed)
# Phase 2: BPE tokenizer, example Swish activation
# Phase 3.5: FAISS context source (requires FAISS library)
# Each C++ plugin is a standalone shared library exporting NNSpire_plugin_descriptor().

# ─── BPE tokenizer plugin ────────────────────────────────────────────────────
# @kb: ai-standards-kb/annexes/A06-Tokenization-Deep-Dive.md
add_library(plugin-bpe-tokenizer SHARED
    bpe_tokenizer/BpeTokenizerPlugin.cpp
)
target_include_directories(plugin-bpe-tokenizer PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)
target_link_libraries(plugin-bpe-tokenizer PRIVATE NNSpire-plugin-api)
set_target_properties(plugin-bpe-tokenizer PROPERTIES
    PREFIX ""   # no lib prefix — dlopen("plugin-bpe-tokenizer.so") works directly
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
target_compile_options(plugin-bpe-tokenizer PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti -fno-exceptions -fvisibility=hidden>
    $<$<CXX_COMPILER_ID:MSVC>:/GR- /EHs->
)

# ─── Example Swish / SiLU activation plugin ──────────────────────────────────
add_library(plugin-example-activation SHARED
    example_activation/ExampleActivationPlugin.cpp
)
target_include_directories(plugin-example-activation PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)
target_link_libraries(plugin-example-activation PRIVATE NNSpire-plugin-api)
set_target_properties(plugin-example-activation PROPERTIES
    PREFIX ""
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
target_compile_options(plugin-example-activation PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti -fno-exceptions -fvisibility=hidden>
    $<$<CXX_COMPILER_ID:MSVC>:/GR- /EHs->
)

# ─── FAISS context source plugin (Phase 3.5 — disabled until FAISS is available)
# @kb: ai-standards-kb/annexes/A02-Vector-Databases.md
if(NN_ENABLE_FAISS)
    find_package(faiss REQUIRED)
    add_library(plugin-faiss-context SHARED
        faiss_context/FaissContextPlugin.cpp
    )
    target_include_directories(plugin-faiss-context PRIVATE "${CMAKE_SOURCE_DIR}/include")
    target_link_libraries(plugin-faiss-context PRIVATE NNSpire-plugin-api faiss)
    set_target_properties(plugin-faiss-context PROPERTIES PREFIX "" CXX_STANDARD 17)
endif()

# Python-side of each built-in plugin lives in python-bridge/NNSpire/plugins/
