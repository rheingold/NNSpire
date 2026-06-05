cmake_minimum_required(VERSION 3.21)

project({{PLUGIN_ID_SNAKE}}
    VERSION {{PLUGIN_VERSION}}
    LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── NNSpire plugin-api headers ─────────────────────────────────────────────
# Set NNSpire_SDK_DIR to the path of your NNSpire installation or source tree.
# Example: cmake -DNNSpire_SDK_DIR=/opt/NNSpire ..
if(NOT DEFINED NNSpire_SDK_DIR)
    message(FATAL_ERROR
        "NNSpire_SDK_DIR is not set. "
        "Run: cmake -DNNSpire_SDK_DIR=<path/to/NNSpire/plugin-api> .."
    )
endif()

# ─── Build shared library ────────────────────────────────────────────────────
add_library({{PLUGIN_ID_SNAKE}} SHARED
    src/{{PLUGIN_ID_SNAKE}}.cpp
)

target_include_directories({{PLUGIN_ID_SNAKE}}
    PRIVATE
        include
        "${NNSpire_SDK_DIR}"
)

# No RTTI or exceptions — matches the NNSpire core ABI
target_compile_options({{PLUGIN_ID_SNAKE}} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti -fno-exceptions>
    $<$<CXX_COMPILER_ID:MSVC>:/GR- /EHs->
)

# Restrict exported symbols to the single required entry point
target_compile_definitions({{PLUGIN_ID_SNAKE}} PRIVATE BUILDING_PLUGIN)

set_target_properties({{PLUGIN_ID_SNAKE}} PROPERTIES
    OUTPUT_NAME "{{PLUGIN_ID_SNAKE}}"
    PREFIX ""
)

# ─── Install ─────────────────────────────────────────────────────────────────
install(TARGETS {{PLUGIN_ID_SNAKE}} DESTINATION .)
