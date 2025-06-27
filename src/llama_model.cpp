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
	
	// Fix: llama_model_default_params() may return -1, reset to 0 for proper detection
	model_params.n_gpu_layers = 0;
	
	// Check for GPU backend availability and provide detailed feedback
	ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
	size_t backend_count = ggml_backend_reg_count();
	bool cuda_backend_registered = false;
	
	UtilityFunctions::print(vformat("🔍 GPU Detection Report:"));
	UtilityFunctions::print(vformat("  - Total backends registered: %d", (int)backend_count));
	
	// Check if CUDA backend is registered
	for (size_t i = 0; i < backend_count; i++) {
		ggml_backend_reg_t reg = ggml_backend_reg_get(i);
		const char* name = ggml_backend_reg_name(reg);
		String backend_name = String(name).to_lower();
		UtilityFunctions::print(vformat("  - Backend %d: %s", (int)i, name));
		if (backend_name.contains("cuda")) {
			cuda_backend_registered = true;
		}
	}
	
	UtilityFunctions::print(vformat("  - CUDA backend registered: %s", cuda_backend_registered ? "YES" : "NO"));
	UtilityFunctions::print(vformat("  - GPU device available: %s", gpu_dev != nullptr ? "YES" : "NO"));
	
	// Configure GPU acceleration if CUDA is available
	if (cuda_backend_registered) {
		UtilityFunctions::print("🚀 CUDA backend found - enabling GPU acceleration");
		// Optimized for 4GB VRAM - conservative layer count to leave room for compute buffers
		model_params.n_gpu_layers = 20; // Use 20/27 layers on GPU, keep 7 on CPU for memory safety
		model_params.main_gpu = 0; // Use first GPU device
		model_params.split_mode = LLAMA_SPLIT_MODE_NONE; // Use single GPU (no splitting)
		UtilityFunctions::print(vformat("  - GPU layers set to: %d", model_params.n_gpu_layers));
		// Optimized memory settings
		model_params.use_mmap = true;
		model_params.use_mlock = false;
		model_params.check_tensors = true;
		UtilityFunctions::print("🚀 CUDA backend detected - GPU acceleration enabled");
		UtilityFunctions::print(vformat("GPU layers: %d/27, VRAM optimized for 4GB", model_params.n_gpu_layers));
	} else {
		UtilityFunctions::print("❌ CUDA backend not available:");
		UtilityFunctions::print("  - Extension was built without CUDA support");
		UtilityFunctions::print("  - Reason: CUDA compiler crashes during build (CUDA 12.0 + complex kernels)");
		UtilityFunctions::print("  - To enable GPU: rebuild with compatible CUDA toolkit version");
		UtilityFunctions::print("  - Alternative: Use OpenCL/ROCm backends if available");
		UtilityFunctions::print("  - Current mode: CPU-only");
		UtilityFunctions::print("  - Performance impact: ~2-10x slower than GPU mode");
		model_params.n_gpu_layers = 0;
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