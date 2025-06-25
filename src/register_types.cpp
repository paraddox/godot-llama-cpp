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
#include "llama_model.h"
#include "llama_model_loader.h"
#include "llama_context.h"

using namespace godot;

static Ref<LlamaModelLoader> llamaModelLoader;

void initialize_types(ModuleInitializationLevel p_level)
{
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	// Initialize llama.cpp backends globally at extension startup
	if (!Engine::get_singleton()->is_editor_hint()) {
		UtilityFunctions::print("Initializing llama.cpp backends...");
		
		// Load backends (CPU should be auto-registered with GGML_USE_CPU)
		ggml_backend_load_all();
		
		// Check if CPU backend is available
		size_t backend_count = ggml_backend_reg_count();
		UtilityFunctions::print(vformat("Backend registrations available: %d", (int)backend_count));
		
		llama_backend_init();
		UtilityFunctions::print("llama.cpp backends initialized");
	}

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