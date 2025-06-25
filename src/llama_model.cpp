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
}

void LlamaModel::load_model() {
	UtilityFunctions::print("load_model: Starting model load process");
	
	if (model) {
		UtilityFunctions::print("load_model: Model already loaded, returning");
		return;
	}

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
	model_params = llama_model_default_params();

	String absPath = ProjectSettings::get_singleton()->globalize_path(get_path());
	UtilityFunctions::print(vformat("load_model: Resolved path: %s", absPath));

	// Log model parameters  
	UtilityFunctions::print(vformat("load_model: Model params - n_gpu_layers: %d", model_params.n_gpu_layers));

	UtilityFunctions::print("load_model: Calling llama_load_model_from_file");
	model = llama_load_model_from_file(absPath.utf8().get_data(), model_params);

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
	return model_params.n_gpu_layers;
}

void LlamaModel::set_n_gpu_layers(int32_t n) {
	model_params.n_gpu_layers = n;
}

LlamaModel::~LlamaModel() {
	if (model) {
		llama_free_model(model);
	}
}