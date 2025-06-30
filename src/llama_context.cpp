#include "llama_context.h"
#include "../llama.cpp/common/common.h"
#include "llama.h"
#include "llama_model.h"
#include "llama_context_pool.h"
#include <algorithm>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/worker_thread_pool.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

void LlamaContext::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_model", "model"), &LlamaContext::set_model);
	ClassDB::bind_method(D_METHOD("get_model"), &LlamaContext::get_model);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::OBJECT, "model", PROPERTY_HINT_RESOURCE_TYPE, "LlamaModel"), "set_model", "get_model");

	ClassDB::bind_method(D_METHOD("get_seed"), &LlamaContext::get_seed);
	ClassDB::bind_method(D_METHOD("set_seed", "seed"), &LlamaContext::set_seed);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");

	ClassDB::bind_method(D_METHOD("get_temperature"), &LlamaContext::get_temperature);
	ClassDB::bind_method(D_METHOD("set_temperature", "temperature"), &LlamaContext::set_temperature);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::FLOAT, "temperature"), "set_temperature", "get_temperature");

	ClassDB::bind_method(D_METHOD("get_top_p"), &LlamaContext::get_top_p);
	ClassDB::bind_method(D_METHOD("set_top_p", "top_p"), &LlamaContext::set_top_p);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::FLOAT, "top_p"), "set_top_p", "get_top_p");

	ClassDB::bind_method(D_METHOD("get_frequency_penalty"), &LlamaContext::get_frequency_penalty);
	ClassDB::bind_method(D_METHOD("set_frequency_penalty", "frequency_penalty"), &LlamaContext::set_frequency_penalty);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::FLOAT, "frequency_penalty"), "set_frequency_penalty", "get_frequency_penalty");

	ClassDB::bind_method(D_METHOD("get_presence_penalty"), &LlamaContext::get_presence_penalty);
	ClassDB::bind_method(D_METHOD("set_presence_penalty", "presence_penalty"), &LlamaContext::set_presence_penalty);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::FLOAT, "presence_penalty"), "set_presence_penalty", "get_presence_penalty");

	ClassDB::bind_method(D_METHOD("get_n_ctx"), &LlamaContext::get_n_ctx);
	ClassDB::bind_method(D_METHOD("set_n_ctx", "n_ctx"), &LlamaContext::set_n_ctx);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::INT, "n_ctx"), "set_n_ctx", "get_n_ctx");

	ClassDB::bind_method(D_METHOD("get_n_len"), &LlamaContext::get_n_len);
	ClassDB::bind_method(D_METHOD("set_n_len", "n_len"), &LlamaContext::set_n_len);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::INT, "n_len"), "set_n_len", "get_n_len");

	ClassDB::bind_method(D_METHOD("initialize_context"), &LlamaContext::initialize_context);
	ClassDB::bind_method(D_METHOD("request_completion", "prompt"), &LlamaContext::request_completion);
	ClassDB::bind_method(D_METHOD("_emit_completion_response", "id", "text", "done"), &LlamaContext::_emit_completion_response);
	ClassDB::bind_method(D_METHOD("_emit_error_response", "id", "error"), &LlamaContext::_emit_error_response);

	ClassDB::bind_method(D_METHOD("set_use_pool", "enabled"), &LlamaContext::set_use_pool);
	ClassDB::bind_method(D_METHOD("get_use_pool"), &LlamaContext::get_use_pool);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::BOOL, "use_pool"), "set_use_pool", "get_use_pool");
	
	ClassDB::bind_method(D_METHOD("set_pool_size", "size"), &LlamaContext::set_pool_size);
	ClassDB::bind_method(D_METHOD("get_pool_size"), &LlamaContext::get_pool_size);
	ClassDB::add_property("LlamaContext", PropertyInfo(Variant::INT, "pool_size"), "set_pool_size", "get_pool_size");

	ADD_SIGNAL(MethodInfo("completion_generated", PropertyInfo(Variant::DICTIONARY, "chunk")));
}

LlamaContext::LlamaContext() {
	ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 4096;
	ctx_params.n_batch = 2048;
	ctx_params.n_ubatch = 512;
	ctx_params.no_perf = false;

	int32_t n_threads = std::min(8, (int)OS::get_singleton()->get_processor_count());
	ctx_params.n_threads = n_threads;
	ctx_params.n_threads_batch = n_threads;

	// Initialize sampling parameters
	temperature = 0.8f;
	top_p = 0.95f;
	penalty_freq = 0.0f;
	penalty_present = 0.0f;
}

void LlamaContext::_enter_tree() {
	// Skip initialization in editor mode
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	// Enable processing to handle response queue
	set_process(true);
	
	// Don't initialize here - wait for explicit call after model is loaded
	// This avoids the race condition where model isn't ready yet
}

void LlamaContext::initialize_context() {
	UtilityFunctions::print("initialize_context: Starting context initialization");
	
	// Skip initialization in editor mode
	if (Engine::get_singleton()->is_editor_hint()) {
		UtilityFunctions::print("initialize_context: In editor mode, skipping");
		return;
	}

	if (model == nullptr) {
		UtilityFunctions::printerr("initialize_context: Failed - model property is null");
		return;
	}

	UtilityFunctions::print("initialize_context: Model property is set, checking if model is loaded");
	if (model->model == NULL) {
		UtilityFunctions::printerr("initialize_context: Failed - model not loaded (model->model is NULL)");
		return;
	}
	
	UtilityFunctions::print("initialize_context: Model is loaded, proceeding with context init");
	
	// Initialize pool if enabled
	if (use_pool) {
		UtilityFunctions::print(vformat("initialize_context: Initializing context pool with %d contexts", pool_size));
		context_pool = new LlamaContextPool();
		
		if (!context_pool->initialize(model, pool_size)) {
			UtilityFunctions::printerr("initialize_context: Failed to initialize context pool");
			delete context_pool;
			context_pool = nullptr;
			use_pool = false; // Fall back to legacy mode
		} else {
			UtilityFunctions::print("initialize_context: Context pool initialized successfully");
			return; // Skip legacy initialization
		}
	}

	// No legacy Godot threading objects needed - using std::thread

	// Backend is already initialized globally, no need to do it again
	// Optimized for single prompt-response usage (no conversation history needed)
	llama_context_params production_params = llama_context_default_params();
	production_params.n_ctx = 1024;   // Good capacity for single prompt-response
	production_params.n_batch = 128;  // Efficient batch size
	production_params.n_ubatch = 128; // Match batch size
	
	// Use multi-threading optimized for system (20 cores available)
	int32_t optimal_threads = std::min(16, (int)OS::get_singleton()->get_processor_count());
	production_params.n_threads = optimal_threads;
	production_params.n_threads_batch = optimal_threads;
	production_params.no_perf = false;  // Enable performance monitoring
	
	UtilityFunctions::print(vformat("initialize_context: Creating production context with n_ctx=%d, n_batch=%d (chunked), n_threads=%d", 
		production_params.n_ctx, production_params.n_batch, production_params.n_threads));
	
	// Verify model is loaded before creating context
	if (model->model == nullptr) {
		UtilityFunctions::printerr("initialize_context: CRITICAL - model->model is nullptr, cannot create context");
		return;
	}
	
	UtilityFunctions::print("initialize_context: Model verified loaded, calling llama_init_from_model...");
	ctx = llama_init_from_model(model->model, production_params);
	UtilityFunctions::print("initialize_context: llama_init_from_model completed successfully");
	if (ctx == NULL) {
		UtilityFunctions::printerr(vformat("%s: Failed to initialize llama context, null ctx", __func__));
		return;
	}

	// Create sampler chain for modern API
	sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
	llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
	llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
	llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));

	UtilityFunctions::print(vformat("%s: Context initialized", __func__));

	// No background thread initialization needed - everything runs on main thread
	UtilityFunctions::print("initialize_context: Single-threaded polling mode enabled");
}

// Main thread processing - called every frame by Godot
void LlamaContext::_process_completions() {
	// Check if there are any new requests
	if (processing_state == IDLE && !request_queue.empty()) {
		current_request = request_queue.front();
		request_queue.pop();
		processing_state = TOKENIZING;
		batch_start_idx = 0;
		generation_complete = false;
		
		UtilityFunctions::print(vformat("Starting completion for request ID: %d", current_request.id));
	}
	
	// Process current request step by step
	switch (processing_state) {
		case IDLE:
			// Nothing to do
			break;
			
		case TOKENIZING: {
			UtilityFunctions::print("Processing: TOKENIZING stage");
			// Safety checks before tokenizing
			if (model.is_null() || !model->model) {
				_emit_completion_response(current_request.id, "Model not loaded", true);
				processing_state = IDLE;
				return;
			}
			
			// Clear KV cache for fresh single-shot context
			if (ctx) {
				llama_kv_self_seq_rm(ctx, 0, -1, -1);
			}
			
			// Tokenize the prompt
			const char* text = current_request.prompt.c_str();
			int text_len = current_request.prompt.length();
			if (text_len == 0) {
				_emit_completion_response(current_request.id, "Empty prompt", true);
				processing_state = IDLE;
				return;
			}
			
			request_tokens.resize(text_len + 16);
			const struct llama_vocab* vocab = llama_model_get_vocab(model->model);
			if (!vocab) {
				_emit_completion_response(current_request.id, "Failed to get vocabulary", true);
				processing_state = IDLE;
				return;
			}
			
			int n_tokens = llama_tokenize(vocab, text, text_len, request_tokens.data(), request_tokens.size(), true, false);
			if (n_tokens < 0) {
				request_tokens.resize(-n_tokens);
				n_tokens = llama_tokenize(vocab, text, text_len, request_tokens.data(), request_tokens.size(), true, false);
			}
			request_tokens.resize(n_tokens);
			
			// For single-shot usage: always start fresh (no shared prefix optimization)
			context_tokens.clear();
			curr_token_pos = 0;
			batch_start_idx = 0;
			processing_state = PROCESSING_BATCH;
			break;
		}
		
		case PROCESSING_BATCH: {
			UtilityFunctions::print(vformat("Processing: PROCESSING_BATCH stage, batch_start_idx=%d, request_tokens.size()=%d", 
				(int)batch_start_idx, (int)request_tokens.size()));
			
			// Safety checks before batch processing
			if (!ctx) {
				_emit_completion_response(current_request.id, "Context not initialized", true);
				processing_state = IDLE;
				return;
			}
			
			// Process one batch per frame
			const int max_batch_size = 32;  // Moderate batches for 1024 context
			if (batch_start_idx < request_tokens.size()) {
				size_t chunk_size = std::min((size_t)max_batch_size, request_tokens.size() - batch_start_idx);
				
				UtilityFunctions::print(vformat("Creating batch: chunk_size=%d, data_offset=%d", 
					(int)chunk_size, (int)batch_start_idx));
				
				if (chunk_size == 0 || request_tokens.empty()) {
					_emit_completion_response(current_request.id, "Invalid batch size", true);
					processing_state = IDLE;
					return;
				}
				
				// Extra safety: check bounds before creating batch
				if (batch_start_idx + chunk_size > request_tokens.size()) {
					_emit_completion_response(current_request.id, "Batch bounds exceeded", true);
					processing_state = IDLE;
					return;
				}
				
				UtilityFunctions::print("Calling llama_batch_get_one...");
				llama_batch chunk_batch = llama_batch_get_one(request_tokens.data() + batch_start_idx, chunk_size);
				UtilityFunctions::print(vformat("Batch created: n_tokens=%d", chunk_batch.n_tokens));
				
				// Validate batch 
				if (chunk_batch.n_tokens <= 0 || chunk_batch.n_tokens != chunk_size) {
					_emit_completion_response(current_request.id, "Invalid batch created", true);
					processing_state = IDLE;
					return;
				}
				
				// Note: llama_batch_get_one() sets logits=nullptr by design
				// We don't need to manually set logits flags for this helper function
				
				bool is_final_chunk = (batch_start_idx + chunk_size >= request_tokens.size());
				
				UtilityFunctions::print("Calling llama_decode...");
				int decode_result = llama_decode(ctx, chunk_batch);
				if (decode_result != 0) {
					// Context is full, complete the processing with error
					UtilityFunctions::print("llama_decode failed (KV cache full), ending completion...");
					_emit_completion_response(current_request.id, "Context memory full", true);
					// Clear context for next single-shot prompt
					context_tokens.clear();
					curr_token_pos = 0;
					processing_state = IDLE;
					return;
				}
				UtilityFunctions::print("llama_decode succeeded");
				
				curr_token_pos += chunk_batch.n_tokens;
				batch_start_idx += chunk_size;
				
				UtilityFunctions::print(vformat("Batch processed: curr_token_pos=%d, new_batch_start_idx=%d", 
					curr_token_pos, (int)batch_start_idx));
				
				if (is_final_chunk) {
					UtilityFunctions::print("Final chunk processed, moving to GENERATING stage");
					context_tokens.insert(context_tokens.end(), request_tokens.begin(), request_tokens.end());
					processing_state = GENERATING;
				}
			} else {
				UtilityFunctions::print("All batches processed, but no final chunk detected - this shouldn't happen");
				processing_state = GENERATING;
			}
			break;
		}
		
		case GENERATING: {
			UtilityFunctions::print(vformat("Processing: GENERATING stage, curr_token_pos=%d", curr_token_pos));
			// Safety checks before generation
			if (!ctx || !sampler || model.is_null() || !model->model) {
				_emit_completion_response(current_request.id, "Generation context not ready", true);
				processing_state = IDLE;
				return;
			}
			
			// Generate one token per frame
			llama_token new_token_id = llama_sampler_sample(sampler, ctx, -1);
			llama_sampler_accept(sampler, new_token_id);
			context_tokens.push_back(new_token_id);
			
			const struct llama_vocab* vocab = llama_model_get_vocab(model->model);
			if (!vocab) {
				_emit_completion_response(current_request.id, "Failed to get vocabulary for generation", true);
				processing_state = IDLE;
				return;
			}
			
			bool eog = llama_vocab_is_eog(vocab, new_token_id);
			bool curr_eq_n_len = curr_token_pos >= n_len;
			
			if (eog || curr_eq_n_len) {
				_emit_completion_response(current_request.id, "", true);
				llama_sampler_reset(sampler);
				// Clear context for next single-shot prompt (no conversation history)
				context_tokens.clear();
				curr_token_pos = 0;
				processing_state = IDLE;
				return;
			}
			
			// Convert token to text
			char token_str[256];
			const struct llama_vocab* token_vocab = llama_model_get_vocab(model->model);
			int len = llama_token_to_piece(token_vocab, new_token_id, token_str, sizeof(token_str), 0, false);
			if (len > 0) {
				token_str[len] = '\0';
				_emit_completion_response(current_request.id, String(token_str), false);
			} else {
				_emit_completion_response(current_request.id, "", false);
			}
			
			// Prepare for next token
			llama_batch new_batch = llama_batch_get_one(&new_token_id, 1);
			curr_token_pos++;
			
			int decode_result = llama_decode(ctx, new_batch);
			if (decode_result != 0) {
				// Context is full, complete the generation
				UtilityFunctions::print("Generation llama_decode failed (context full), ending generation...");
				_emit_completion_response(current_request.id, "", true);
				llama_sampler_reset(sampler);
				// Clear context for next single-shot prompt
				context_tokens.clear();
				curr_token_pos = 0;
				processing_state = IDLE;
				return;
			}
			break;
		}
	}
}

PackedStringArray LlamaContext::_get_configuration_warnings() const {
	PackedStringArray warnings;
	if (model == NULL) {
		warnings.push_back("Model resource property not defined");
	}
	return warnings;
}

// Main thread processing - called every frame by Godot
void LlamaContext::_process(double delta) {
	_process_completions();
}

// Thread-safe helper functions (called from main thread via call_deferred)
void LlamaContext::_emit_completion_response(int id, const String& text, bool done) {
	Dictionary response;
	response["id"] = id;
	response["text"] = text;
	response["done"] = done;
	emit_signal("completion_generated", response);
}

void LlamaContext::_emit_error_response(int id, const String& error) {
	Dictionary response;
	response["id"] = id;
	response["error"] = error;
	emit_signal("completion_generated", response);
}

int LlamaContext::request_completion(const String &prompt) {
	// Use pool if available
	if (use_pool && context_pool) {
		return context_pool->submit_request(prompt, temperature, top_p, n_len);
	}
	
	// Check if context is properly initialized
	if (!ctx) {
		UtilityFunctions::printerr(vformat("%s: Context not initialized - llama_init_from_model failed (ctx is null)", __func__));
		return -1;
	}

	int id = request_id++;

	UtilityFunctions::print(vformat("%s: Requesting completion for prompt id: %d", __func__, id));

	// Queue request using simple main-thread queue
	completion_request req = { id, std::string(prompt.utf8().get_data()) };
	request_queue.push(req);

	return id;
}

void LlamaContext::set_model(const Ref<LlamaModel> p_model) {
	model = p_model;
}
Ref<LlamaModel> LlamaContext::get_model() {
	return model;
}

uint32_t LlamaContext::get_seed() {
	// Note: seed is no longer part of context params in modern API
	// Return a placeholder value
	return 0;
}
void LlamaContext::set_seed(uint32_t seed) {
	// Note: seed handling moved to model/sampler level in modern API
	// This is kept for compatibility but doesn't do anything
}

uint32_t LlamaContext::get_n_ctx() {
	return ctx_params.n_ctx;
}
void LlamaContext::set_n_ctx(uint32_t n_ctx) {
	ctx_params.n_ctx = n_ctx;
}

int32_t LlamaContext::get_n_len() {
	return n_len;
}
void LlamaContext::set_n_len(int32_t n_len) {
	this->n_len = n_len;
}

float LlamaContext::get_temperature() {
	return temperature;
}
void LlamaContext::set_temperature(float temp) {
	this->temperature = temp;
	// Note: Sampler chain will be updated on next completion request
}

float LlamaContext::get_top_p() {
	return top_p;
}
void LlamaContext::set_top_p(float p) {
	this->top_p = p;
	// Note: Sampler chain will be updated on next completion request
}

float LlamaContext::get_frequency_penalty() {
	return penalty_freq;
}
void LlamaContext::set_frequency_penalty(float frequency_penalty) {
	this->penalty_freq = frequency_penalty;
	// Note: Sampler chain will be updated on next completion request
}

float LlamaContext::get_presence_penalty() {
	return penalty_present;
}
void LlamaContext::set_presence_penalty(float presence_penalty) {
	this->penalty_present = presence_penalty;
	// Note: Sampler chain will be updated on next completion request
}

void LlamaContext::_exit_tree() {
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	// Cleanup pool if active
	if (context_pool) {
		context_pool->shutdown_pool();
		delete context_pool;
		context_pool = nullptr;
	}

	// Reset processing state
	processing_state = IDLE;

	if (ctx) {
		llama_free(ctx);
	}
	if (sampler) {
		llama_sampler_free(sampler);
		sampler = nullptr;
	}
	llama_backend_free();
}

// Pool management methods
void LlamaContext::set_use_pool(bool enabled) {
	use_pool = enabled;
}

bool LlamaContext::get_use_pool() const {
	return use_pool;
}

void LlamaContext::set_pool_size(uint32_t size) {
	pool_size = std::max(1u, std::min(16u, size)); // Clamp between 1-16
}

uint32_t LlamaContext::get_pool_size() const {
	return pool_size;
}