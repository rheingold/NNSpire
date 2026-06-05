# Python pybind11 bridge — LGPL v3
# Exposes NNSpire-builtin (and transitively NNSpire-core) to Python as the
# `NNSpire` package with a torch-compatible naming convention.
#
# Build requirements:
#   • Python 3.10+ (Interpreter + Development headers)
#   • pybind11 ≥ 2.11 (found via pip site-packages or FetchContent — see main CMakeLists.txt)
#
# Enable with:
#   cmake --preset engine-ninja -DNN_ENABLE_PYTHON=ON

pybind11_add_module(NNSpire_pyd
    bindings/module.cpp
)

target_include_directories(NNSpire_pyd PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(NNSpire_pyd PRIVATE
    NNSpire-builtin
    NNSpire-backend-cpu
)

# pybind11 requires C++ exceptions — do NOT disable them here.
# On MinGW + MSVC-compiled Python, pybind11 headers mix MSVC and GCC
# exception-handling conventions; -fpermissive downgrades those one-line
# ordering warnings to non-errors so the build succeeds.
target_compile_options(NNSpire_pyd PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-fpermissive>
)

set_target_properties(NNSpire_pyd PROPERTIES
    OUTPUT_NAME "NNSpire"     # import NNSpire
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/NNSpire"
)

# Copy pure-Python package files alongside the compiled .pyd/.so
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/NNSpire/__init__.py"
    "${CMAKE_BINARY_DIR}/python/NNSpire/__init__.py"
    COPYONLY
)

message(STATUS "Python bridge: pybind11 NNSpire module configured")
