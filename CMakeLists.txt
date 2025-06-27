cmake_minimum_required(VERSION 3.17)
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
set(EXTENSION_NAME "godot-llama-cpp-${ARCH_NAME}-${PLATFORM_NAME}-gnu-${CMAKE_BUILD_TYPE}")

# Output directory
if(NOT DEFINED ADDON_OUTPUT_DIR)
    set(ADDON_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/godot/addons/godot-llama-cpp" CACHE PATH "Output directory for the addon")
endif()

#
# Phase 1: Build llama.cpp using official CMake
#
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")

# GPU acceleration (OFF for compatibility testing)
set(GGML_CUDA OFF CACHE BOOL "Disable CUDA for compatibility testing")

# Force Position Independent Code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(llama.cpp)

#
# Phase 2: Build godot-cpp using OFFICIAL method
#
set(GODOT_PROJECT_DIR "${CMAKE_SOURCE_DIR}/godot" CACHE STRING "")
add_subdirectory(godot_cpp)

#
# Phase 3: Build our extension using OFFICIAL patterns
#
file(GLOB_RECURSE EXTENSION_SOURCES "src/*.cpp" "src/*.c")

add_library(${EXTENSION_NAME} SHARED ${EXTENSION_SOURCES})

# CRITICAL: Use exact same target linking as official test
target_link_libraries(${EXTENSION_NAME} PRIVATE
    godot-cpp::template_release  # Use official godot-cpp target!
    llama
    ggml
)

# CRITICAL: Use exact same compiler flags as official test
target_compile_features(${EXTENSION_NAME} PUBLIC cxx_std_17)

target_compile_options(${EXTENSION_NAME} PRIVATE
    # CRITICAL: Disable exceptions (matches official build)
    -fno-exceptions
    # CRITICAL: Symbol visibility
    -fvisibility=hidden
    # CRITICAL: Position independent code
    -fPIC
)

# CRITICAL: Use exact same properties as official test
set_target_properties(${EXTENSION_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    
    POSITION_INDEPENDENT_CODE ON
    BUILD_RPATH_USE_ORIGIN ON
    
    # Note: Removed aggressive static linking that was causing libgcc_s linking issues
    
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${ADDON_OUTPUT_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${ADDON_OUTPUT_DIR}/lib"
)

target_include_directories(${EXTENSION_NAME} PRIVATE
    src
    llama.cpp/include
    llama.cpp/src
    llama.cpp/common
    llama.cpp/ggml/include
    llama.cpp/ggml/src
)

# Platform-specific settings
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(${EXTENSION_NAME} PRIVATE _GNU_SOURCE)
    target_link_libraries(${EXTENSION_NAME} PRIVATE pthread dl)
endif()

# CRITICAL: Success message
add_custom_command(TARGET ${EXTENSION_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo 
            "✅ FIXED Extension built: ${ADDON_OUTPUT_DIR}/lib/$<TARGET_FILE_NAME:${EXTENSION_NAME}>"
    COMMENT "Build complete with official patterns"
)