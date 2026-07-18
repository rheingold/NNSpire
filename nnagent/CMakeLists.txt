cmake_minimum_required(VERSION 3.20)
project(nnagent_core VERSION 0.1.0 LANGUAGES CXX)

# ─── C++ Standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Build Type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ─── Output Directories ────────────────────────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ─── Core Library ──────────────────────────────────────────────────────────────
add_library(nnagent_core STATIC
  core/src/config/ConfigLoader.cpp
  core/src/config/ConfigValidator.cpp
  core/src/logging/Logger.cpp
  core/src/core/AppShell.cpp
)

target_include_directories(nnagent_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_definitions(nnagent_core PRIVATE
  NNAgent_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  NNAgent_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  NNAgent_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# ─── External Dependencies (nlohmann/json via FetchContent) ────────────────────
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

target_link_libraries(nnagent_core PUBLIC nlohmann_json::nlohmann_json)

# ─── Tests ─────────────────────────────────────────────────────────────────────
option(NNAgent_BUILD_TESTS "Build unit and integration tests" ON)

if(NNAgent_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ─── CLI (optional) ────────────────────────────────────────────────────────────
option(NNAgent_BUILD_CLI "Build CLI executable" OFF)

if(NNAgent_BUILD_CLI)
  add_subdirectory(cli)
endif()

# ─── Install (PH1 — simplified, full export in PH2) ────────────────────────────
install(TARGETS nnagent_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY core/include/nnagent
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Also install from core/include for the corrected source layout
install(DIRECTORY core/include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
