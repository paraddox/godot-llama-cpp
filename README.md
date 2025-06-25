<div align='center'>

<img width="100" src="/godot/addons/godot-llama-cpp/assets/godot-llama-cpp-1024x1024.svg">

<h1>godot-llama-cpp</h1>

Run large language models in [Godot](https://godotengine.org). Powered by [llama.cpp](https://github.com/ggerganov/llama.cpp).

<br />
<br />

![Godot v4.2+](https://img.shields.io/badge/Godot-v4.2%2B-%23478cbf?logo=godot-engine&logoColor=white)
![Build System](https://img.shields.io/badge/Build-CMake-%23064F8C?logo=cmake&logoColor=white)
![GitHub last commit](https://img.shields.io/github/last-commit/hazelnutcloud/godot-llama-cpp)
![GitHub License](https://img.shields.io/github/license/hazelnutcloud/godot-llama-cpp)

</div>

## Overview

This library aims to provide a high-level interface to run large language models in Godot, following Godot's node-based design principles.

```gdscript
@onready var llama_context = %LlamaContext

var messages = [
  { "sender": "system", "text": "You are a pirate chatbot who always responds in pirate speak!" },
  { "sender": "user", "text": "Who are you?" }
]
var prompt = ChatFormatter.apply("llama3", messages)
var completion_id = llama_context.request_completion(prompt)

while (true):
  var response = await llama_context.completion_generated
  print(response["text"])

  if response["done"]: break
```

## Features
  - **🎯 Stable CMake Build System** - Migrated from Zig for better compatibility
  - **🔧 Modern llama.cpp Integration** - Uses official build configuration
  - **⚡ Configurable Deployment** - Flexible addon installation paths
  - **🔄 Asynchronous Completion Generation** - Non-blocking text generation
  - **📦 GGUF Resource Support** - Model files as native Godot resources
  - **🌐 Cross-Platform Compatibility** - Support for multiple operating systems
  
  ### Platform & Compute Backend Support:
  | Platform | CPU | Metal | Vulkan | CUDA |
  |----------|-----|-------|--------|------|
  | macOS    | ✅  | ✅    | ❌     | ❌   |
  | Linux    | ✅  | ❌    | ✅     | 🚧   |
  | Windows  | ✅  | ❌    | 🚧     | 🚧   |

## Roadmap
  - [ ] Chat completions support with better template handling
  - [ ] Grammar support (JSON, structured output)
  - [ ] Multimodal models support (vision, audio)
  - [ ] Embeddings generation
  - [ ] Vector database integration
  - [ ] Streaming performance optimizations
  - [ ] GPU backend improvements (CUDA, Vulkan)

## Building & Installation

> **🎯 NEW: Now uses CMake build system for better stability and compatibility!**

### Prerequisites
- **CMake** (3.14 or higher)
- **Python 3** (for binding generation)
- **C++ compiler** with C++17 support
- **Git** (for submodules)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone --recurse-submodules https://github.com/hazelnutcloud/godot-llama-cpp.git
   cd godot-llama-cpp
   ```

2. **Build the extension:**
   ```bash
   # For default demo project
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   
   # OR for your custom project
   mkdir build && cd build
   cmake -DADDON_OUTPUT_DIR=/path/to/your-project/addons/godot-llama-cpp ..
   make -j$(nproc)
   ```

3. **Copy addon to your project** (if not using custom output):
   ```bash
   cp -r godot/addons/godot-llama-cpp /path/to/your-project/addons/
   ```

4. **Enable the plugin** in your Godot project settings.

5. **Add the `LlamaContext` node** to your scene.

6. **Run your project** and enjoy!

### Build Examples

```bash
# Build for current demo project
mkdir build && cd build
cmake .. && make -j$(nproc)

# Build for external project
mkdir build && cd build
cmake -DADDON_OUTPUT_DIR=$HOME/my-game/addons/godot-llama-cpp ..
make -j$(nproc)

# Build Debug version (larger but with symbols)
cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j$(nproc)
```

### Legacy Zig Build (Deprecated)

The old Zig build system has been replaced due to SIGILL crashes. Use CMake instead:
```bash
# ❌ OLD (causes crashes)
zig build --prefix <project>/addons/godot-llama-cpp

# ✅ NEW (stable)
cmake -DADDON_OUTPUT_DIR=<project>/addons/godot-llama-cpp .. && make
```

## Troubleshooting

### Common Issues

**Extension fails to load:**
- Ensure both Debug and Release versions are built
- Check that the `.gdextension` file paths match your built libraries
- Verify CMake build completed without errors

**Build failures:**
- Install CMake 3.14+ and Python 3
- Ensure submodules are initialized: `git submodule update --init --recursive`
- Check that you have a C++17 compatible compiler

**Performance issues:**
- Use Release build for production: `cmake -DCMAKE_BUILD_TYPE=Release ..`
- Consider GPU backends for supported platforms
- Ensure model files are in GGUF format

**SIGILL crashes (old Zig builds):**
- Migrate to CMake build system (this resolves CPU instruction compatibility issues)
- Remove old Zig-built libraries and rebuild with CMake

### Getting Help

- Check [CLAUDE.md](CLAUDE.md) for detailed technical documentation
- Report issues on the GitHub repository
- Include build logs and system information when reporting bugs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
