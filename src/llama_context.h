#ifndef LLAMA_CONTEXT_H
#define LLAMA_CONTEXT_H

#include "llama.h"
#include "../llama.cpp/common/common.h"
#include "llama_model.h"
#include "llama_context_pool.h"
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <functional>
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/semaphore.hpp>
#include <godot_cpp/classes/thread.hpp>
#include <godot_cpp/templates/vector.hpp>
#include <godot_cpp/core/class_db.hpp>
namespace godot {

struct completion_request {
	int id;
	std::string prompt;  // Use std::string to avoid Godot String in background thread
};

struct completion_response {
	int id;
	std::string text;  // Use std::string to avoid Godot String in background thread
	bool done;
};

// Single-threaded completion processing - all on main thread

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

	// Single-threaded polling architecture (NO background threads)
	std::queue<completion_request> request_queue;
	std::vector<llama_token> context_tokens;
	
	// Current processing state
	enum ProcessingState {
		IDLE,
		TOKENIZING,
		PROCESSING_BATCH,
		GENERATING
	} processing_state = IDLE;
	
	// Current request being processed
	completion_request current_request;
	std::vector<llama_token> request_tokens;
	size_t batch_start_idx = 0;
	int curr_token_pos = 0;
	bool generation_complete = false;
  
  // New pool-based architecture
  LlamaContextPool* context_pool = nullptr;
  bool use_pool = false;  // Disable pooling by default to prevent CUDA OOM
  uint32_t pool_size = 1; // Use single context to minimize memory usage

protected:
	static void _bind_methods();

public:
	void set_model(const Ref<LlamaModel> model);
	Ref<LlamaModel> get_model();
	
	void initialize_context();
	int request_completion(const String &prompt);
	
	// Main thread processing (called periodically by Godot)
	void _process_completions();
	
	// Thread-safe helper functions (called from main thread)
	void _emit_completion_response(int id, const String& text, bool done);
	void _emit_error_response(int id, const String& error);
	
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
	virtual void _process(double delta) override;
  virtual void _exit_tree() override;
	LlamaContext();
};
} //namespace godot

#endif