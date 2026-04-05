# nnstudio-plugin-api — C ABI headers, TrustVerifier, TrustUpdateHandler, PluginLoader
# Phase 2
#
# Layout mirrors core/ and builtin/ — headers under include/plugin-api/,
# sources under src/plugin-api/.  This subdirectory owns only CMake logic.
#
#   include/plugin-api/nnstudio_plugin.h        ← C ABI for plugin authors
#   include/plugin-api/PluginLoader.h
#   include/plugin-api/trust/TrustStore.h
#   include/plugin-api/trust/TrustVerifier.h
#   include/plugin-api/trust/TrustUpdateHandler.h
#   src/plugin-api/trust/TrustStore.cpp
#   src/plugin-api/trust/TrustVerifier.cpp      (NN_ENABLE_TRUST only)
#   src/plugin-api/trust/TrustUpdateHandler.cpp (NN_ENABLE_TRUST only)
#   src/plugin-api/PluginLoader.cpp             (NN_ENABLE_TRUST only)

# Interface-only target — plugin authors include this; no compiled code needed
# for the C ABI contract itself.
add_library(nnstudio-plugin-api INTERFACE)
target_include_directories(nnstudio-plugin-api INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

# ─── nnstudio-trust: compiled trust + plugin loading code (OpenSSL required) ──
if(NN_ENABLE_TRUST)
    set(PLUGIN_API_SOURCES
        ${CMAKE_SOURCE_DIR}/src/plugin-api/trust/TrustStore.cpp
        ${CMAKE_SOURCE_DIR}/src/plugin-api/trust/TrustVerifier.cpp
        ${CMAKE_SOURCE_DIR}/src/plugin-api/trust/TrustUpdateHandler.cpp
        ${CMAKE_SOURCE_DIR}/src/plugin-api/PluginLoader.cpp
    )
    add_library(nnstudio-trust STATIC ${PLUGIN_API_SOURCES})
    target_include_directories(nnstudio-trust
        PUBLIC  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
                "$<INSTALL_INTERFACE:include>"
    )
    target_link_libraries(nnstudio-trust
        PUBLIC  nnstudio-plugin-api nnstudio-core
        PRIVATE OpenSSL::SSL OpenSSL::Crypto
    )
    target_compile_options(nnstudio-trust PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti -fno-exceptions>
        $<$<CXX_COMPILER_ID:MSVC>:/GR- /EHs->
    )

    # ─── nnstudio-sign — developer CLI (keygen, sign, verify, submit, create-tup) ──
    add_executable(nnstudio-sign
        ${CMAKE_SOURCE_DIR}/src/plugin-api/tools/sign/main.cpp
    )
    target_link_libraries(nnstudio-sign PRIVATE nnstudio-trust)
    # nnstudio-sign IS allowed exceptions (user-facing CLI tool)
    target_compile_options(nnstudio-sign PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-fno-rtti>
        $<$<CXX_COMPILER_ID:MSVC>:/GR->
    )

else()
    # No-op INTERFACE stub: everything that links nnstudio-trust still compiles.
    add_library(nnstudio-trust INTERFACE)
    target_include_directories(nnstudio-trust INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
    )
endif()
