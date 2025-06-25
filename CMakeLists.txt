cmake_minimum_required(VERSION 3.14)

project(godot-llama-cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform detection
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(PLATFORM_NAME "linux")
    set(ARCH_NAME "x86_64")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(PLATFORM_NAME "macos")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(ARCH_NAME "arm64")
    else()
        set(ARCH_NAME "x86_64")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(PLATFORM_NAME "windows")
    set(ARCH_NAME "x86_64")
endif()

# Extension naming
string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE_LOWER)
set(EXTENSION_NAME "libgodot-llama-cpp-${ARCH_NAME}-${PLATFORM_NAME}-gnu-${CMAKE_BUILD_TYPE}")

# Configurable output directory for final extension (equivalent to Zig's --prefix)
if(NOT DEFINED ADDON_OUTPUT_DIR)
    set(ADDON_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/godot/addons/godot-llama-cpp" CACHE PATH "Output directory for the addon")
endif()

#
# Phase 1: llama.cpp as subdirectory (official CMake build)
#

# Configure llama.cpp options for our use case
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")  # We want static libs to embed

# Force Position Independent Code for all targets (required for shared library linking)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add llama.cpp - this creates 'ggml' and 'llama' targets
add_subdirectory(llama.cpp)

#
# Phase 2: Generate Godot bindings and build godot-cpp
#

# Generate bindings first
find_package(Python3 REQUIRED)
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/gen/include/godot_cpp/classes/global_constants.hpp
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/binding_generator.py 
            ${CMAKE_SOURCE_DIR}/godot_cpp/gdextension/extension_api.json
            ${CMAKE_BINARY_DIR}
    DEPENDS ${CMAKE_SOURCE_DIR}/binding_generator.py 
            ${CMAKE_SOURCE_DIR}/godot_cpp/gdextension/extension_api.json
    COMMENT "Generating Godot C++ bindings"
)

# Custom target to ensure bindings are generated
add_custom_target(generate_bindings
    DEPENDS ${CMAKE_BINARY_DIR}/gen/include/godot_cpp/classes/global_constants.hpp
)

# Collect godot-cpp source files
file(GLOB_RECURSE GODOT_CPP_SOURCES 
    "godot_cpp/src/*.cpp"
    "godot_cpp/src/*.c"
)

# Create godot-cpp static library
add_library(godot-cpp STATIC ${GODOT_CPP_SOURCES})
add_dependencies(godot-cpp generate_bindings)

target_include_directories(godot-cpp PRIVATE
    godot_cpp/gdextension
    godot_cpp/include
    ${CMAKE_BINARY_DIR}/gen/include
)

target_compile_options(godot-cpp PRIVATE
    -std=c++17
    -fno-exceptions
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(godot-cpp PRIVATE DEBUG_ENABLED)
endif()

#
# Phase 3: Build our extension
#

# Collect our source files
file(GLOB_RECURSE EXTENSION_SOURCES "src/*.cpp" "src/*.c")

# Create the main extension shared library
add_library(${EXTENSION_NAME} SHARED ${EXTENSION_SOURCES})

# Link against llama.cpp's official targets and our godot-cpp
target_link_libraries(${EXTENSION_NAME} PRIVATE
    llama      # Official llama.cpp target
    ggml       # Official ggml target  
    godot-cpp  # Our godot-cpp build
)

target_include_directories(${EXTENSION_NAME} PRIVATE
    src
    godot_cpp/gdextension
    godot_cpp/include
    ${CMAKE_BINARY_DIR}/gen/include
    llama.cpp/include
    llama.cpp/src
    llama.cpp/common
    llama.cpp/ggml/include
    llama.cpp/ggml/src
)

# Platform-specific settings
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(${EXTENSION_NAME} PRIVATE
        _GNU_SOURCE
    )
    # Link required system libraries
    target_link_libraries(${EXTENSION_NAME} PRIVATE pthread dl)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_compile_definitions(${EXTENSION_NAME} PRIVATE
        _DARWIN_C_SOURCE
    )
endif()

# Use C++17 standard
target_compile_features(${EXTENSION_NAME} PRIVATE cxx_std_17)

#
# Phase 4: Installation
#

# Create addon directory structure
add_custom_command(TARGET ${EXTENSION_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory 
            ${ADDON_OUTPUT_DIR}/lib
    COMMENT "Creating addon directory structure"
)

# Copy built extension to addon directory
add_custom_command(TARGET ${EXTENSION_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy 
            $<TARGET_FILE:${EXTENSION_NAME}>
            ${ADDON_OUTPUT_DIR}/lib/${EXTENSION_NAME}.so
    COMMENT "Installing extension to addon directory"
)

# Print success message
add_custom_command(TARGET ${EXTENSION_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo 
            "✅ Extension built successfully: ${ADDON_OUTPUT_DIR}/lib/${EXTENSION_NAME}.so"
    COMMENT "Build complete"
)