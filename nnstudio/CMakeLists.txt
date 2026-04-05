cmake_minimum_required(VERSION 3.21)

project(NNStudio
    VERSION 0.1.0
    DESCRIPTION "Neural Network design, training, deployment and learning workbench"
    HOMEPAGE_URL "https://github.com/rheingold/NNStudio"
    LANGUAGES CXX C
)

# ─── C++ standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Options ─────────────────────────────────────────────────────────────────
option(NN_QT_VERSION          "Qt major version to target (5 or 6)"   6)
option(NN_ENABLE_APP          "Build Qt GUI application"              OFF)
option(NN_ENABLE_CUDA         "Build CUDA backend"                    OFF)
option(NN_ENABLE_PYTHON       "Build Python pybind11 bridge"          OFF)
option(NN_ENABLE_TRUST        "Build trust/PKI subsystem (OpenSSL)"   OFF)
option(NN_ENABLE_TESTS        "Build unit tests"                      ON)
option(NN_ENABLE_RUNNER       "Build nnstudio-runner sidecar"         OFF)
option(NN_STATIC_CORE         "Build nnstudio-core as static lib"     ON)

# ─── CMake module path ───────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ─── Find Qt (only when building the GUI app) ───────────────────────────────
if(NN_ENABLE_APP)
    if(NN_QT_VERSION EQUAL 6)
        find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Qml Quick Charts
                                             Network Concurrent WebEngineWidgets)
        qt_standard_project_setup()
    else()
        find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets Qml Quick Charts
                                             Network Concurrent WebEngineWidgets)
    endif()
endif()

# ─── OpenSSL (only when building trust/PKI subsystem) ────────────────────────
if(NN_ENABLE_TRUST)
    find_package(OpenSSL 3.0 REQUIRED)
endif()

find_package(Threads REQUIRED)

if(NN_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(NN_ENABLE_PYTHON)
    find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 REQUIRED
                 HINTS "${CMAKE_CURRENT_SOURCE_DIR}/third-party-deps/pybind11")
endif()

# ─── Third-party: Eigen (header-only linear algebra) ─────────────────────────
# Use local copy if already in third-party/, otherwise fetch from upstream.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third-party-deps/eigen/Eigen/Dense")
    message(STATUS "Eigen: using third-party-deps/eigen")
    add_library(eigen INTERFACE)
    target_include_directories(eigen INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/third-party-deps/eigen")
else()
    message(STATUS "Eigen: fetching via FetchContent (3.4.0)")
    include(FetchContent)
    FetchContent_Declare(Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
    )
    set(EIGEN_BUILD_DOC        OFF CACHE BOOL "")
    set(EIGEN_BUILD_PKGCONFIG  OFF CACHE BOOL "")
    set(BUILD_TESTING          OFF CACHE BOOL "")
    FetchContent_MakeAvailable(Eigen)
    # Eigen's own CMakeLists already creates the 'eigen' INTERFACE target
endif()

# ─── Subdirectories ──────────────────────────────────────────────────────────

# ─── nnstudio-core ───────────────────────────────────────────────────────────────
# Namespace: nnstudio::core::   Headers: include/core/   Sources: src/core/
set(CORE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/core/Tensor.cpp
    ${CMAKE_SOURCE_DIR}/src/core/BackendRegistry.cpp
    ${CMAKE_SOURCE_DIR}/src/core/Layer.cpp
    ${CMAKE_SOURCE_DIR}/src/core/Trainer.cpp
    # future:
    # ${CMAKE_SOURCE_DIR}/src/internal/training/Trainer.cpp
    # ${CMAKE_SOURCE_DIR}/src/internal/graph/ComputeGraph.cpp
    # ${CMAKE_SOURCE_DIR}/src/internal/formats/FormatIO.cpp
)

if(NN_STATIC_CORE)
    add_library(nnstudio-core STATIC ${CORE_SOURCES})
else()
    add_library(nnstudio-core SHARED ${CORE_SOURCES})
endif()

target_include_directories(nnstudio-core
    PUBLIC  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:include>"
    PRIVATE "${CMAKE_SOURCE_DIR}/src"
)
target_link_libraries(nnstudio-core
    PUBLIC  eigen
    PRIVATE Threads::Threads
            $<$<BOOL:${NN_ENABLE_TRUST}>:OpenSSL::Crypto>
)
target_compile_options(nnstudio-core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti -fno-exceptions>
    $<$<CXX_COMPILER_ID:MSVC>:/GR- /EHs->
)

# ─── nnstudio-backend-cpu ─────────────────────────────────────────────────────────
# Namespace: nnstudio::builtin::backends::   Source: src/builtin/backends/
if(NN_STATIC_CORE)
    add_library(nnstudio-backend-cpu STATIC
        ${CMAKE_SOURCE_DIR}/src/builtin/backends/CpuBackend.cpp)
else()
    add_library(nnstudio-backend-cpu SHARED
        ${CMAKE_SOURCE_DIR}/src/builtin/backends/CpuBackend.cpp)
endif()
target_include_directories(nnstudio-backend-cpu
    PUBLIC  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(nnstudio-backend-cpu PUBLIC nnstudio-core PRIVATE eigen)
target_compile_options(nnstudio-backend-cpu PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti -fno-exceptions>
    $<$<CXX_COMPILER_ID:MSVC>:/GR- /EHs->
)

# ─── nnstudio-builtin (layers / losses / optimizers) ────────────────────────────
# Namespace: nnstudio::builtin::   Headers: include/builtin/   Sources: src/builtin/
set(BUILTIN_SOURCES
    ${CMAKE_SOURCE_DIR}/src/builtin/layers/Activations.cpp
    ${CMAKE_SOURCE_DIR}/src/builtin/layers/Dense.cpp
    ${CMAKE_SOURCE_DIR}/src/builtin/losses/Losses.cpp
    ${CMAKE_SOURCE_DIR}/src/builtin/optimizers/Optimizers.cpp
)
if(NN_STATIC_CORE)
    add_library(nnstudio-builtin STATIC ${BUILTIN_SOURCES})
else()
    add_library(nnstudio-builtin SHARED ${BUILTIN_SOURCES})
endif()
target_include_directories(nnstudio-builtin
    PUBLIC  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(nnstudio-builtin
    PUBLIC  nnstudio-core
    PRIVATE nnstudio-backend-cpu eigen
)
target_compile_options(nnstudio-builtin PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti -fno-exceptions>
    $<$<CXX_COMPILER_ID:MSVC>:/GR- /EHs->
)

# ─── nnstudio-backend-cuda (Phase 1, NN_ENABLE_CUDA) ─────────────────────────────
if(NN_ENABLE_CUDA)
    add_library(nnstudio-backend-cuda SHARED
        # ${CMAKE_SOURCE_DIR}/src/builtin/backends/CudaBackend.cpp
        # ${CMAKE_SOURCE_DIR}/src/builtin/backends/CudaBackend.cu
    )
    set_target_properties(nnstudio-backend-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(nnstudio-backend-cuda
        PUBLIC  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
                "$<INSTALL_INTERFACE:include>"
    )
    target_link_libraries(nnstudio-backend-cuda
        PUBLIC  nnstudio-core
        PRIVATE CUDA::cublas CUDA::cudnn
    )
endif()

# ─── nnstudio-backend-quantum stub (Phase 6) ─────────────────────────────────────
add_library(nnstudio-backend-quantum INTERFACE)
target_include_directories(nnstudio-backend-quantum INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>")
target_link_libraries(nnstudio-backend-quantum INTERFACE nnstudio-core)

add_subdirectory(plugin-api)   # always — provides nnstudio-trust INTERFACE stub

# Built-in plugins: Phase 2 — skip until plugin ABI is stable
# add_subdirectory(plugins)

# Pipeline, deployment, app: require Qt — Phase 3+
if(NN_ENABLE_APP)
    add_subdirectory(pipeline)
    add_subdirectory(deployment)
    add_subdirectory(app)
endif()

if(NN_ENABLE_PYTHON)
    add_subdirectory(python-bridge)
endif()

if(NN_ENABLE_RUNNER)
    add_subdirectory(runner)
endif()

# ─── Tests ───────────────────────────────────────────────────────────────────
if(NN_ENABLE_TESTS)
    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third-party-deps/googletest/CMakeLists.txt")
        message(STATUS "GTest: using third-party-deps/googletest")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third-party-deps/googletest"
                         "${CMAKE_BINARY_DIR}/googletest")
    else()
        message(STATUS "GTest: fetching via FetchContent (v1.14.0)")
        include(FetchContent)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    add_subdirectory(tests)
endif()

# ─── Summary ─────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "NNStudio ${PROJECT_VERSION}")
message(STATUS "  App (Qt)     : ${NN_ENABLE_APP}")
message(STATUS "  Trust (PKI)  : ${NN_ENABLE_TRUST}")
message(STATUS "  CUDA backend : ${NN_ENABLE_CUDA}")
message(STATUS "  Python bridge: ${NN_ENABLE_PYTHON}")
message(STATUS "  Tests        : ${NN_ENABLE_TESTS}")
message(STATUS "  Runner       : ${NN_ENABLE_RUNNER}")
message(STATUS "")
