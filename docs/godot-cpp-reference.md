# Godot C++ GDExtension Reference

This file contains API reference documentation for Godot C++ bindings (godot-cpp) used in this project.

## Project Overview

godot-cpp provides C++ bindings for Godot's GDExtension API, allowing you to extend Godot with C++ code that integrates seamlessly with the engine.

Key features:
- Nearly the same level of access as statically linked C++ modules
- Object-oriented C++ API wrapping the C GDExtension interface
- Automatic binding generation from extension API
- Full access to Godot's class hierarchy and signals

## Version Compatibility

⚠️ **IMPORTANT**: Use the godot-cpp branch that matches your Godot version:
- `master` - Development branch for next Godot 4.x minor release
- `4.4`, `4.3`, `4.2`, `4.1`, `4.0` - Stable release branches
- Git tags (e.g., `godot-4.1.1-stable`) for specific versions

**Compatibility Rules**:
- Extensions targeting earlier versions work in later minor versions
- Extensions targeting later versions won't work in earlier versions
- **Exception**: Extensions for Godot 4.0 won't work in 4.1+

## Core Concepts

### GDExtension Structure

A typical GDExtension project structure:
```
your_project/
├── demo/                    # Test Godot project
├── godot-cpp/              # C++ bindings (submodule)
├── src/                    # Your C++ source code
│   ├── your_class.h
│   ├── your_class.cpp
│   └── register_types.cpp
└── SConstruct              # SCons build file
```

### GDCLASS Macro

Every custom class must use the `GDCLASS` macro:
```cpp
#include <godot_cpp/classes/node.hpp>

namespace godot {
    class MyNode : public Node {
        GDCLASS(MyNode, Node)
        
    private:
        double time_passed;
        
    protected:
        static void _bind_methods();
        
    public:
        MyNode();
        ~MyNode();
        
        void _process(double delta) override;
    };
}
```

### Method Binding

The `_bind_methods()` static function registers methods and properties:
```cpp
void MyNode::_bind_methods() {
    // Bind methods
    ClassDB::bind_method(D_METHOD("get_time"), &MyNode::get_time);
    ClassDB::bind_method(D_METHOD("set_time", "time"), &MyNode::set_time);
    
    // Add properties
    ClassDB::add_property("MyNode", 
        PropertyInfo(Variant::FLOAT, "time"), 
        "set_time", "get_time");
    
    // Add signals
    ADD_SIGNAL(MethodInfo("time_changed", 
        PropertyInfo(Variant::FLOAT, "new_time")));
}
```

## Key Classes and Headers

### Base Classes
```cpp
#include <godot_cpp/classes/node.hpp>           // Node base class
#include <godot_cpp/classes/resource.hpp>       // Resource base class
#include <godot_cpp/classes/ref_counted.hpp>    // RefCounted base class
#include <godot_cpp/classes/object.hpp>         // Object base class
```

### Core Functionality
```cpp
#include <godot_cpp/core/class_db.hpp>          // Class registration
#include <godot_cpp/core/defs.hpp>              // Core definitions
#include <godot_cpp/godot.hpp>                  // Main GDExtension header
#include <gdextension_interface.h>              // C interface definitions
```

### Variant System
```cpp
#include <godot_cpp/variant/variant.hpp>        // Variant type
#include <godot_cpp/variant/dictionary.hpp>     // Dictionary
#include <godot_cpp/variant/array.hpp>          // Array
#include <godot_cpp/variant/string.hpp>         // String
#include <godot_cpp/variant/utility_functions.hpp> // Utility functions
```

### Engine Integration
```cpp
#include <godot_cpp/classes/engine.hpp>         // Engine singleton
#include <godot_cpp/classes/os.hpp>             // OS singleton
#include <godot_cpp/classes/worker_thread_pool.hpp> // Threading
```

## Registration and Initialization

### Module Registration Pattern
```cpp
// register_types.h
#ifndef REGISTER_TYPES_H
#define REGISTER_TYPES_H

#include <godot_cpp/core/class_db.hpp>

using namespace godot;

void initialize_my_module(ModuleInitializationLevel p_level);
void uninitialize_my_module(ModuleInitializationLevel p_level);

#endif
```

```cpp
// register_types.cpp
#include "register_types.h"
#include "my_class.h"

#include <gdextension_interface.h>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

void initialize_my_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    
    GDREGISTER_RUNTIME_CLASS(MyClass);
}

void uninitialize_my_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
}

extern "C" {
    GDExtensionBool GDE_EXPORT my_library_init(
        GDExtensionInterfaceGetProcAddress p_get_proc_address,
        const GDExtensionClassLibraryPtr p_library,
        GDExtensionInitialization *r_initialization) {
        
        godot::GDExtensionBinding::InitObject init_obj(
            p_get_proc_address, p_library, r_initialization);
        
        init_obj.register_initializer(initialize_my_module);
        init_obj.register_terminator(uninitialize_my_module);
        init_obj.set_minimum_library_initialization_level(
            MODULE_INITIALIZATION_LEVEL_SCENE);
        
        return init_obj.init();
    }
}
```

## Property and Method Binding

### Property Registration
```cpp
void MyClass::_bind_methods() {
    // Simple property
    ClassDB::bind_method(D_METHOD("get_value"), &MyClass::get_value);
    ClassDB::bind_method(D_METHOD("set_value", "value"), &MyClass::set_value);
    ClassDB::add_property("MyClass", 
        PropertyInfo(Variant::INT, "value"), 
        "set_value", "get_value");
    
    // Property with hints
    ClassDB::add_property("MyClass",
        PropertyInfo(Variant::FLOAT, "speed", PROPERTY_HINT_RANGE, "0,100,0.1"),
        "set_speed", "get_speed");
    
    // Resource property
    ClassDB::add_property("MyClass",
        PropertyInfo(Variant::OBJECT, "resource", 
                    PROPERTY_HINT_RESOURCE_TYPE, "Resource"),
        "set_resource", "get_resource");
}
```

### Method Registration
```cpp
void MyClass::_bind_methods() {
    // Method with no parameters
    ClassDB::bind_method(D_METHOD("reset"), &MyClass::reset);
    
    // Method with parameters
    ClassDB::bind_method(D_METHOD("move", "direction", "speed"), 
                        &MyClass::move);
    
    // Method with default values
    ClassDB::bind_method(D_METHOD("jump", "height"), &MyClass::jump, 
                        DEFVAL(10.0));
    
    // Virtual method (override from parent)
    ClassDB::bind_method(D_METHOD("_ready"), &MyClass::_ready);
}
```

### Signal Registration
```cpp
void MyClass::_bind_methods() {
    // Signal with no parameters
    ADD_SIGNAL(MethodInfo("finished"));
    
    // Signal with parameters
    ADD_SIGNAL(MethodInfo("value_changed", 
        PropertyInfo(Variant::INT, "old_value"),
        PropertyInfo(Variant::INT, "new_value")));
}

// Emit signals in your code
void MyClass::some_method() {
    emit_signal("finished");
    emit_signal("value_changed", old_val, new_val);
}
```

## Threading and Synchronization

### Worker Thread Pool
```cpp
#include <godot_cpp/classes/worker_thread_pool.hpp>

void MyClass::start_background_task() {
    WorkerThreadPool::TaskID task = WorkerThreadPool::get_singleton()
        ->add_task(callable_mp(this, &MyClass::background_work), false);
}

void MyClass::background_work() {
    // Background processing
    // Use call_thread_safe for main thread communication
    call_thread_safe("_on_background_complete");
}
```

### Thread-Safe Communication
```cpp
void MyClass::background_thread_method() {
    // Process data in background
    
    // Safely call main thread method
    call_thread_safe("_update_ui", result_data);
    
    // Safely emit signal
    call_thread_safe("emit_signal", "data_ready", result);
}
```

## Memory Management

### Reference Counting
```cpp
// For RefCounted objects, use Ref<T>
Ref<MyResource> resource;
resource.instantiate();  // Create new instance
resource->some_method();

// Automatic cleanup when Ref goes out of scope
```

### Manual Memory Management
```cpp
// For Node objects, add to scene tree for automatic cleanup
MyNode* node = memnew(MyNode);
add_child(node);  // Node will be cleaned up with parent

// Or manually manage
MyNode* node = memnew(MyNode);
// ... use node ...
memdelete(node);  // Manual cleanup
```

## Integration with Godot Systems

### Engine Integration
```cpp
void MyClass::_enter_tree() {
    // Don't run in editor
    if (Engine::get_singleton()->is_editor_hint()) {
        return;
    }
    
    // Initialize your extension
    setup_resources();
}

void MyClass::_exit_tree() {
    cleanup_resources();
}
```

### Utility Functions
```cpp
#include <godot_cpp/variant/utility_functions.hpp>

void MyClass::debug_output() {
    UtilityFunctions::print("Debug message");
    UtilityFunctions::printerr("Error message");
    UtilityFunctions::print(vformat("Value: %d", some_value));
}
```

## Build Configuration

### .gdextension File
```ini
[configuration]
entry_symbol = "my_library_init"
compatibility_minimum = "4.1"

[libraries]
macos.debug = "res://bin/libmyext.macos.debug.framework"
macos.release = "res://bin/libmyext.macos.release.framework"
windows.debug.x86_64 = "res://bin/libmyext.windows.debug.x86_64.dll"
windows.release.x86_64 = "res://bin/libmyext.windows.release.x86_64.dll"
linux.debug.x86_64 = "res://bin/libmyext.linux.debug.x86_64.so"
linux.release.x86_64 = "res://bin/libmyext.linux.release.x86_64.so"
```

### SCons Build
```python
#!/usr/bin/env python
import os
import sys

env = SConscript("godot-cpp/SConstruct")

# Define source files
sources = Glob("src/*.cpp")

# Create shared library
if env["platform"] == "macos":
    library = env.SharedLibrary(
        "demo/bin/libmyext.{}.{}.framework/libmyext.{}.{}".format(
            env["platform"], env["target"], env["platform"], env["target"]
        ),
        source=sources,
    )
else:
    library = env.SharedLibrary(
        "demo/bin/libmyext{}{}".format(env["suffix"], env["SHLIBSUFFIX"]),
        source=sources,
    )

Default(library)
```

## Common Patterns

### Resource Management
```cpp
class MyResource : public Resource {
    GDCLASS(MyResource, Resource)
    
public:
    void load_from_file(const String& path) {
        FileAccess* file = FileAccess::open(path, FileAccess::READ);
        if (file) {
            // Read data
            file->close();
            memdelete(file);
        }
    }
};
```

### Singleton Pattern
```cpp
class MyManager : public Object {
    GDCLASS(MyManager, Object)
    
private:
    static MyManager* instance;
    
public:
    static MyManager* get_singleton() { return instance; }
    
    MyManager() { instance = this; }
    ~MyManager() { instance = nullptr; }
};
```

## Migration Notes

When updating godot-cpp versions:
1. Check the branch/tag compatibility
2. Update build scripts if needed
3. Test all registered classes and methods
4. Update .gdextension compatibility_minimum
5. Rebuild bindings with new extension_api.json if needed

## Links

- [Official GDExtension Documentation](https://docs.godotengine.org/en/stable/tutorials/scripting/gdextension/index.html)
- [godot-cpp Repository](https://github.com/godotengine/godot-cpp)
- [GDExtension Template](https://github.com/godotengine/godot-cpp-template)
- [API Class Reference](https://docs.godotengine.org/en/stable/classes/index.html)