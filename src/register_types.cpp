#include "register_types.h"
#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <llama.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#ifdef GGML_USE_CUDA
#include <ggml-cuda.h>
#endif
#include "llama_model.h"
#include "llama_model_loader.h"
#include "llama_context.h"

using namespace godot;

static Ref<LlamaModelLoader> llamaModelLoader;

void initialize_types(ModuleInitializationLevel p_level)
{
	UtilityFunctions::print(vformat("🚀 initialize_types called with level: %d", (int)p_level));
	
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		UtilityFunctions::print("Skipping - not SCENE level");
		return;
	}
	
	UtilityFunctions::print("✅ Extension initialization starting...");

	// Initialize llama.cpp backends globally at extension startup
	// Note: Allow initialization in editor for testing and development
	UtilityFunctions::print("Initializing llama.cpp backends...");
	
	// Preprocessor test - this should always show a message
#ifdef GGML_USE_CUDA
	UtilityFunctions::print("🔧 Preprocessor test: GGML_USE_CUDA is DEFINED");
#else  
	UtilityFunctions::print("🔧 Preprocessor test: GGML_USE_CUDA is NOT DEFINED");
#endif
	
	// Load backends (CPU and GPU backends should be auto-registered)
	ggml_backend_load_all();
	
	// Force registration of CUDA backend if available
#ifdef GGML_USE_CUDA
	UtilityFunctions::print("Manually registering CUDA backend...");
	ggml_backend_reg_t cuda_reg = ggml_backend_cuda_reg();
	if (cuda_reg) {
		UtilityFunctions::print("✅ CUDA backend registration successful");
	} else {
		UtilityFunctions::print("❌ CUDA backend registration failed");
	}
#endif
	
	// Debug: Check if GGML_USE_CUDA is defined
#ifdef GGML_USE_CUDA
	UtilityFunctions::print("✅ GGML_USE_CUDA is defined - checking CUDA availability...");
#else
	UtilityFunctions::print("❌ GGML_USE_CUDA is NOT defined - CUDA support disabled");
#endif

	// Explicitly try to register CUDA backend if available
#ifdef GGML_USE_CUDA
	UtilityFunctions::print("✅ GGML_USE_CUDA is defined - checking CUDA availability...");
	int cuda_device_count = ggml_backend_cuda_get_device_count();
	UtilityFunctions::print(vformat("ggml_backend_cuda_get_device_count() returned: %d", cuda_device_count));
	if (cuda_device_count > 0) {
		UtilityFunctions::print(vformat("🚀 CUDA devices detected: %d", cuda_device_count));
		
		// Get CUDA device info
		for (int i = 0; i < cuda_device_count; i++) {
			char desc[256];
			ggml_backend_cuda_get_device_description(i, desc, sizeof(desc));
			UtilityFunctions::print(vformat("  Device %d: %s", i, String(desc)));
		}
		
		 // Force register CUDA backend for all devices
		for (int i = 0; i < cuda_device_count; i++) {
			ggml_backend_reg_t cuda_reg = ggml_backend_cuda_reg();
			if (cuda_reg) {
				UtilityFunctions::print(vformat("Force registering CUDA backend for device %d", i));
			}
		}
	} else {
		UtilityFunctions::print("❌ No CUDA devices detected by ggml_backend_cuda_get_device_count()");
	}
#else
	UtilityFunctions::print("⚠️  GGML_USE_CUDA not defined - CUDA support not compiled");
#endif
		
		// Detailed backend detection and reporting
		size_t backend_count = ggml_backend_reg_count();
		UtilityFunctions::print(vformat("Total backend registrations available: %d", (int)backend_count));
		
		// Check for specific backend types
		bool has_cpu = false, has_cuda = false, has_vulkan = false;
		
		for (size_t i = 0; i < backend_count; i++) {
			ggml_backend_reg_t reg = ggml_backend_reg_get(i);
			const char* name = ggml_backend_reg_name(reg);
			UtilityFunctions::print(vformat("  Backend %d: %s", (int)i, String(name)));
			
			String backend_name = String(name).to_lower();
			if (backend_name.contains("cpu")) has_cpu = true;
			else if (backend_name.contains("cuda")) has_cuda = true;
			else if (backend_name.contains("vulkan")) has_vulkan = true;
		}
		
		// Report acceleration status
		if (has_cuda) {
			UtilityFunctions::print("🚀 GPU Acceleration: CUDA backend detected - high performance expected!");
		} else if (has_vulkan) {
			UtilityFunctions::print("🚀 GPU Acceleration: Vulkan backend detected - good performance expected!");
		} else if (has_cpu) {
			UtilityFunctions::print("⚠️  CPU-Only: No GPU backends detected - performance may be limited");
		} else {
			UtilityFunctions::printerr("❌ No backends detected - check build configuration");
		}
		
	llama_backend_init();
	UtilityFunctions::print("llama.cpp backends initialized successfully");

	ClassDB::register_class<LlamaModelLoader>();
	llamaModelLoader.instantiate();
	ResourceLoader::get_singleton()->add_resource_format_loader(llamaModelLoader);

	ClassDB::register_class<LlamaModel>();
  ClassDB::register_class<LlamaContext>();
}

void uninitialize_types(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	ResourceLoader::get_singleton()->remove_resource_format_loader(llamaModelLoader);
	llamaModelLoader.unref();
}

extern "C"
{
	// Initialization
	GDExtensionBool GDE_EXPORT init_library(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization)
	{
		GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);
		init_obj.register_initializer(initialize_types);
		init_obj.register_terminator(uninitialize_types);
		init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

		return init_obj.init();
	}
}