#ifndef LLAMA_MODEL_H
#define LLAMA_MODEL_H

#include <llama.h>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>

namespace godot {

class LlamaModel : public Resource {
	GDCLASS(LlamaModel, Resource);

private:
	llama_model_params model_params;

protected:
	static void _bind_methods();

public:
	llama_model *model = nullptr;
	void load_model();
	bool is_loaded();

	int32_t get_n_gpu_layers();
	void set_n_gpu_layers(int32_t n);

	LlamaModel();
	~LlamaModel();
};

} //namespace godot

#endif