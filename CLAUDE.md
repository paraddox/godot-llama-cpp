# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

🎯 **This project now uses CMake build system (migrated from Zig to resolve SIGILL crashes):**

**CMake Build (Recommended):**
- `mkdir build && cd build` - Create build directory
- `cmake ..` - Configure CMake build 
- `make -j$(nproc)` - Build the plugin shared library
- Built extension automatically copied to `godot/addons/godot-llama-cpp/lib/`

**Legacy Zig Build (Deprecated):**
- `zig build --prefix <path>` - Old Zig build (causes SIGILL crashes)
- Use CMake instead for stable builds

### API Migration Status

✅ **COMPLETED: This project has been fully adapted to modern llama.cpp and godot-cpp versions**

The project has been successfully updated to work with the latest API versions and is now fully functional.

**Current Status:**
- ✅ Build system updated for modern llama.cpp file structure
- ✅ Submodules updated to latest versions  
- ✅ API compatibility restored (all major API calls updated)
- ✅ Modern sampling API implementation with sampler chains
- ✅ Modern tokenization API using vocab-based functions
- ✅ GDCLASS compatibility fixed for modern godot-cpp
- ✅ Modern batch API implementation
- ✅ **Backend initialization fixed - extension loads successfully in Godot 4.4**
- ✅ **Runtime crashes resolved - llama.cpp models load properly**

**llama.cpp API Changes Applied:**
- ✅ `llama_token_is_eog(model, token)` → `llama_vocab_is_eog(llama_model_get_vocab(model), token)`
- ✅ `llama_new_context_with_model()` → `llama_init_from_model()` 
- ✅ `llama_kv_cache_seq_rm()` → Removed (functionality simplified in batching)
- ✅ `llama_tokenize(ctx, text, tokens, n_max_tokens)` → `llama_tokenize(vocab, text, text_len, tokens, n_max_tokens, add_special, parse_special)`
- ✅ `llama_token_to_piece(model, token)` → `llama_token_to_piece(vocab, token, buf, length, lstrip, special)`
- ✅ `llama_sampling_context` → `llama_sampler` with sampler chains
- ✅ `llama_batch_clear/add()` → `llama_batch_get_one()` for modern batch API
- ✅ Context params seed handling updated

**godot-cpp API Changes Applied:**
- ✅ `GDCLASS(LlamaContext, Node)` → `GDCLASS(LlamaContext, Node);` (semicolon added)
- ✅ Added required `#include <godot_cpp/core/class_db.hpp>` headers
- ✅ Updated binding generation compatible with Godot 4.2+

**Build Dependencies:**
- ✅ Removed: `json-schema-to-grammar.cpp` (nlohmann/json dependency removed)
- ✅ Updated: All file paths for modern llama.cpp structure
- ✅ Updated: `llama.cpp/ggml/src/ggml.c` (was in root)
- ✅ Updated: `llama.cpp/src/llama.cpp` (was in root)

**Critical Fixes Applied:**
- ✅ **Backend Initialization Order**: Moved `ggml_backend_load_all()` and `llama_backend_init()` to global extension initialization in `register_types.cpp`
- ✅ **GGML_USE_CPU Define**: Added essential `-DGGML_USE_CPU` compile flag to enable automatic CPU backend registration
- ✅ **Resource Loading Race Condition**: Added backend availability check to prevent model loading before backends are ready
- ✅ **Extension Loading**: Fixed crashes during Godot editor startup by properly handling editor mode vs runtime mode
- ✅ **Context Initialization**: Separated context initialization from `_enter_tree()` to avoid race conditions with model loading
- ✅ **Model Loading Sequence**: Ensured proper initialization order: backend defines → automatic registration → model loading
- ✅ **SIGILL Crash RESOLVED**: Migrated from custom Zig build to official llama.cpp CMake build system
- 🎯 **CMake Migration SUCCESS**: Now uses llama.cpp's proven build configuration with PIC linking
- 🔧 **Technical Resolution**: Official CMake handles CPU instruction compatibility automatically

**Recovery Options:**

1. **Pin to Compatible Versions (Recommended for quick fix):**
   ```bash
   cd llama.cpp && git checkout cb49e0f8c906e5da49e9f6d64a57742a9a241c6a
   cd ../godot_cpp && git checkout f002ca18  # Early 2024 version
   ```

2. **Full API Migration (Comprehensive solution):**
   - Update all llama.cpp API calls to new signatures
   - Fix godot-cpp GDCLASS compatibility
   - Add missing nlohmann/json dependency
   - Update sampling API usage

3. **Hybrid Approach:**
   - Use llama.cpp's official CMake build instead of custom Zig build
   - Create minimal Zig wrapper for Godot integration only

### Troubleshooting

**Common Build Errors:**

1. **`llama_token_is_eog` not found:**
   - Update to `llama_vocab_is_eog(llama_model_get_vocab(model), token)`

2. **`nlohmann/json_fwd.hpp` not found:**
   - Remove `json-schema-to-grammar.cpp` from build or install nlohmann-json

3. **`GDCLASS` initialization errors:**
   - Indicates godot-cpp version mismatch - use compatible version

4. **`llama_sampling_context` undefined:**
   - Sampling API completely restructured in newer llama.cpp

5. **`llama_new_context_with_model` deprecated:**
   - Replace with `llama_init_from_model` and update parameter handling

**Development Workflow:**
1. Start with pinned compatible versions to get basic functionality
2. Gradually update individual components
3. Test each API change incrementally
4. Consider using llama.cpp examples as reference for modern API usage

### Build Options

- `--Dcompute-backend=<backend>` - Choose compute backend: `cpu` (default), `metal`, `vulkan`, or `cuda`
- Metal backend requires macOS with Metal frameworks
- Vulkan backend requires `VULKAN_SDK` environment variable

## Development Setup

1. Clone with submodules: `git clone --recurse-submodules <repo-url>`
2. Install dependencies: Zig v0.13.0, Python (for binding generation)
3. For Vulkan support: Set `VULKAN_SDK` environment variable
4. Copy addon to Godot project: `cp -r godot/addons/godot-llama-cpp <project>/addons/`
5. Build extension: `zig build --prefix <project>/addons/godot-llama-cpp`

## Architecture

### Core Components

- **LlamaModel** (`src/llama_model.*`) - Godot Resource wrapper for llama.cpp models, handles GGUF files as Godot resources
- **LlamaContext** (`src/llama_context.*`) - Main Node for LLM inference, manages threading and completion requests
- **LlamaModelLoader** (`src/llama_model_loader.*`) - Handles loading GGUF model files

### Key Integration Points

- `src/register_types.*` - Registers classes with Godot's ClassDB
- `binding_generator.py` - Generates C++ bindings from Godot extension API
- `godot/addons/godot-llama-cpp/` - Godot plugin structure with GDScript helpers

### Threading Model

LlamaContext uses Godot's threading system:
- Main thread handles requests via `request_completion()`
- Background thread (`__thread_loop()`) processes completions
- Thread-safe communication using Mutex and Semaphore

### Chat System

- `ChatFormatter` class provides chat template formatting for different models (Llama3, Phi3, Mistral)
- Message format: `{"sender": "user/system/assistant", "text": "content"}`
- Streaming responses via `completion_generated` signal

## Project Structure

- `src/` - C++ extension source code
- `godot/` - Godot project with plugin and examples
- `godot_cpp/` - Godot-cpp submodule for C++ bindings
- `llama.cpp/` - llama.cpp submodule for LLM inference
- `tools/` - Build utilities (concat files, Metal shader expansion)

## Testing

Run the example project:
1. Open `godot/` directory in Godot 4.2+
2. Enable the "godot-llama-cpp" plugin in project settings
3. Run the simple example scene

## Platform Support

| Platform | CPU | Metal | Vulkan | CUDA |
|----------|-----|-------|--------|------|
| macOS    | ✅  | ✅    | ❌     | ❌   |
| Linux    | ✅  | ❌    | ✅     | 🚧   |
| Windows  | ✅  | ❌    | 🚧     | 🚧   |