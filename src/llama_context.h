#ifndef LLAMA_CONTEXT_H
#define LLAMA_CONTEXT_H

#include "llama.h"
#include "../llama.cpp/common/common.h"
#include "llama_model.h"
#include "llama_context_pool.h"
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/semaphore.hpp>
#include <godot_cpp/classes/thread.hpp>
#include <godot_cpp/templates/vector.hpp>
#include <godot_cpp/core/class_db.hpp>
namespace godot {

struct completion_request {
	int id;
	String prompt;
};

class LlamaContext : public Node {
	GDCLASS(LlamaContext, Node);

private:
	Ref<LlamaModel> model;
	// Legacy single-context fields (kept for compatibility)
	llama_context *ctx = nullptr;
  struct llama_sampler *sampler = nullptr;
	llama_context_params ctx_params;
  // Sampling parameters now stored individually
  float temperature = 0.8f;
  float top_p = 0.95f;
  float penalty_freq = 0.0f;
  float penalty_present = 0.0f;
  int32_t n_len = 1024;
	int request_id = 0;
	Vector<completion_request> completion_requests;

	// Legacy threading (kept for compatibility)
	Ref<Thread> thread;
	Ref<Semaphore> semaphore;
	Ref<Mutex> mutex;
  std::vector<llama_token> context_tokens;
  bool exit_thread = false;
  
  // New pool-based architecture
  LlamaContextPool* context_pool = nullptr;
  bool use_pool = true;
  uint32_t pool_size = 4;

protected:
	static void _bind_methods();

public:
	void set_model(const Ref<LlamaModel> model);
	Ref<LlamaModel> get_model();
	
	void initialize_context();
	int request_completion(const String &prompt);
	void __thread_loop();
	
	// Pool management
	void set_use_pool(bool enabled);
	bool get_use_pool() const;
	void set_pool_size(uint32_t size);
	uint32_t get_pool_size() const;

	uint32_t get_seed();
	void set_seed(uint32_t seed);
	uint32_t get_n_ctx();
	void set_n_ctx(uint32_t n_ctx);
  int32_t get_n_len();
  void set_n_len(int32_t n_len);
  float get_temperature();
  void set_temperature(float temperature);
  float get_top_p();
  void set_top_p(float top_p);
  float get_frequency_penalty();
  void set_frequency_penalty(float frequency_penalty);
  float get_presence_penalty();
  void set_presence_penalty(float presence_penalty);

	virtual PackedStringArray _get_configuration_warnings() const override;
	virtual void _enter_tree() override;
  virtual void _exit_tree() override;
	LlamaContext();
};
} //namespace godot

#endif