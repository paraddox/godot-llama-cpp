#include "llama_model.h"
#include "llama.h"
#include "ggml-cpu.h"
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/engine.hpp>

using namespace godot;

void LlamaModel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_model"), &LlamaModel::load_model);
	ClassDB::bind_method(D_METHOD("is_loaded"), &LlamaModel::is_loaded);

	ClassDB::bind_method(D_METHOD("get_n_gpu_layers"), &LlamaModel::get_n_gpu_layers);
	ClassDB::bind_method(D_METHOD("set_n_gpu_layers", "n"), &LlamaModel::set_n_gpu_layers);
	ClassDB::add_property("LlamaModel", PropertyInfo(Variant::INT, "n_gpu_layers"), "set_n_gpu_layers", "get_n_gpu_layers");
}

LlamaModel::LlamaModel() {
	// Don't initialize model params in constructor to avoid calling llama.cpp functions before backend loading
	// Initialize model to nullptr
	model = nullptr;
	printf("LlamaModel::LlamaModel() constructor called\n");
	UtilityFunctions::print("LlamaModel::LlamaModel() constructor called");
}

void LlamaModel::load_model() {
	UtilityFunctions::print("load_model: Starting model load process");
	
	if (model) {
		UtilityFunctions::print("load_model: Model already loaded, returning");
		return;
	}
	
	UtilityFunctions::print(vformat("load_model: Current model_params.n_gpu_layers: %d (before initialization)", model_params.n_gpu_layers));

	if (Engine::get_singleton()->is_editor_hint()) {
		UtilityFunctions::print("load_model: In editor mode, skipping");
		return;
	}

	UtilityFunctions::print("load_model: Checking backend availability");
	// Check if backends are available by trying to get CPU backend device
	ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
	if (cpu_dev == nullptr) {
		UtilityFunctions::printerr("load_model: CPU backend device not available, skipping model load");
		return;
	}
	UtilityFunctions::print("load_model: CPU backend device found");

	UtilityFunctions::print("load_model: Initializing model parameters");
	// Initialize model parameters (backends are loaded globally)
	UtilityFunctions::print("load_model: About to call llama_model_default_params()");
	model_params = llama_model_default_params();
	UtilityFunctions::print(vformat("load_model: Default params - n_gpu_layers: %d", model_params.n_gpu_layers));
	
	// Auto-configure GPU acceleration - attempt GPU even if runtime detection fails
	// Check both backend registration and runtime device availability
	ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
	size_t backend_count = ggml_backend_reg_count();
	bool cuda_backend_registered = false;
	
	// Check if CUDA backend is registered (even if runtime detection fails)
	for (size_t i = 0; i < backend_count; i++) {
		ggml_backend_reg_t reg = ggml_backend_reg_get(i);
		const char* name = ggml_backend_reg_name(reg);
		String backend_name = String(name).to_lower();
		if (backend_name.contains("cuda")) {
			cuda_backend_registered = true;
			break;
		}
	}
	
	// Force GPU acceleration if CUDA is available
	if (cuda_backend_registered) {
		// Force enable GPU layers - override any existing setting
		UtilityFunctions::print(vformat("load_model: Before setting GPU layers: %d", model_params.n_gpu_layers));
		model_params.n_gpu_layers = -1; // Use all available GPU layers
		UtilityFunctions::print(vformat("load_model: After setting GPU layers: %d", model_params.n_gpu_layers));
		UtilityFunctions::print("🚀 CUDA backend detected - forcing GPU acceleration (all layers)");
		UtilityFunctions::print("    Note: If GPU is busy with graphics, model will gracefully fall back to CPU");
	} else {
		UtilityFunctions::print("⚠️ No GPU backend available - using CPU-only mode");
		model_params.n_gpu_layers = 0;
		UtilityFunctions::print(vformat("load_model: Set to CPU mode - GPU layers: %d", model_params.n_gpu_layers));
	}

	String absPath = ProjectSettings::get_singleton()->globalize_path(get_path());
	UtilityFunctions::print(vformat("load_model: Resolved path: %s", absPath));

	// Log model parameters in detail 
	UtilityFunctions::print(vformat("load_model: Model params - n_gpu_layers: %d", model_params.n_gpu_layers));
	UtilityFunctions::print(vformat("load_model: Model params - use_mmap: %s", model_params.use_mmap ? "true" : "false"));
	UtilityFunctions::print(vformat("load_model: Model params - use_mlock: %s", model_params.use_mlock ? "true" : "false"));
	UtilityFunctions::print(vformat("load_model: Model params address: %p", &model_params));

	UtilityFunctions::print("load_model: Calling llama_model_load_from_file");
	UtilityFunctions::print(vformat("load_model: Passing params with n_gpu_layers: %d", model_params.n_gpu_layers));
	
	// CRITICAL DEBUG: Print to both printf and Godot to catch this call
	printf("=== CRITICAL: llama_model_load_from_file called with n_gpu_layers: %d ===\n", model_params.n_gpu_layers);
	fflush(stdout);
	
	model = llama_model_load_from_file(absPath.utf8().get_data(), model_params);
	UtilityFunctions::print("load_model: llama_model_load_from_file returned");

	if (model == NULL) {
		UtilityFunctions::printerr(vformat("load_model: llama_load_model_from_file returned NULL for path: %s", absPath));
		return;
	}

	UtilityFunctions::print(vformat("load_model: SUCCESS - Model loaded from %s", __func__, absPath));
}

bool LlamaModel::is_loaded() {
	return model != nullptr;
}

int32_t LlamaModel::get_n_gpu_layers() {
	printf("get_n_gpu_layers() called, returning: %d\n", model_params.n_gpu_layers);
	return model_params.n_gpu_layers;
}

void LlamaModel::set_n_gpu_layers(int32_t n) {
	printf("set_n_gpu_layers() called with: %d\n", n);
	model_params.n_gpu_layers = n;
	printf("set_n_gpu_layers() set model_params.n_gpu_layers to: %d\n", model_params.n_gpu_layers);
}

LlamaModel::~LlamaModel() {
	if (model) {
		llama_model_free(model);
	}
}