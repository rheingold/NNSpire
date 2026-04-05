# nnstudio-plugin-api — C ABI headers, TrustVerifier, TrustUpdateHandler, PluginLoader
# Phase 2

# Interface-only target for plugin authors (C ABI headers only, no compiled code needed)
add_library(nnstudio-plugin-api INTERFACE)
target_include_directories(nnstudio-plugin-api INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

# Compiled code: TrustVerifier, TrustUpdateHandler, PluginLoader, TrustStore
# Only built when NN_ENABLE_TRUST=ON (requires OpenSSL)
if(NN_ENABLE_TRUST)
    set(PLUGIN_API_SOURCES
        # trust/TrustStore.cpp
        # trust/TrustVerifier.cpp
        # trust/TrustUpdateHandler.cpp
        # PluginLoader.cpp
    )
    add_library(nnstudio-trust STATIC ${PLUGIN_API_SOURCES})
    target_include_directories(nnstudio-trust
        PUBLIC  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(nnstudio-trust
        PUBLIC  nnstudio-plugin-api
        PRIVATE OpenSSL::SSL OpenSSL::Crypto
    )

    # nnstudio-sign CLI tool
    add_executable(nnstudio-sign
        # tools/sign/main.cpp
    )
    target_link_libraries(nnstudio-sign PRIVATE nnstudio-trust)
else()
    # Provide a no-op INTERFACE stub so all targets that link nnstudio-trust compile
    add_library(nnstudio-trust INTERFACE)
    target_include_directories(nnstudio-trust INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    )
endif()
