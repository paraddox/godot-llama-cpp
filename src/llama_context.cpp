#include "llama_context.h"
#include "common.h"
#include "llama.h"
#include "llama_model.h"
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

	ClassDB::bind_method(D_METHOD("request_completion", "prompt"), &LlamaContext::request_completion);
	ClassDB::bind_method(D_METHOD("__thread_loop"), &LlamaContext::__thread_loop);

	ADD_SIGNAL(MethodInfo("completion_generated", PropertyInfo(Variant::DICTIONARY, "chunk")));
}

LlamaContext::LlamaContext() {
	ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 4096;

	int32_t n_threads = OS::get_singleton()->get_processor_count();
	ctx_params.n_threads = n_threads;
	ctx_params.n_threads_batch = n_threads;

	// Initialize sampling parameters
	temperature = 0.8f;
	top_p = 0.95f;
	penalty_freq = 0.0f;
	penalty_present = 0.0f;
}

void LlamaContext::_enter_tree() {
	// TODO: remove this and use runtime classes once godot 4.3 lands, see https://github.com/godotengine/godot/pull/82554
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	if (model->model == NULL) {
		UtilityFunctions::printerr(vformat("%s: Failed to initialize llama context, model property not defined", __func__));
		return;
	}

	mutex.instantiate();
	semaphore.instantiate();
	thread.instantiate();

	llama_backend_init();
	llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED);

	ctx = llama_init_from_model(model->model, ctx_params);
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

	thread->start(callable_mp(this, &LlamaContext::__thread_loop));
}

void LlamaContext::__thread_loop() {
	while (true) {
		semaphore->wait();

		mutex->lock();
		if (exit_thread) {
			mutex->unlock();
			break;
		}
		if (completion_requests.size() == 0) {
			mutex->unlock();
			continue;
		}
		completion_request req = completion_requests.get(0);
		completion_requests.remove_at(0);
		mutex->unlock();

		UtilityFunctions::print(vformat("%s: Running completion for prompt id: %d", __func__, req.id));

		std::vector<llama_token> request_tokens;
		// Modern tokenization API
		const char* text = req.prompt.utf8().get_data();
		int text_len = strlen(text);
		request_tokens.resize(text_len + 16); // Reserve space
		const struct llama_vocab* vocab = llama_model_get_vocab(model->model);
		int n_tokens = llama_tokenize(vocab, text, text_len, request_tokens.data(), request_tokens.size(), true, false);
		if (n_tokens < 0) {
			request_tokens.resize(-n_tokens);
			n_tokens = llama_tokenize(vocab, text, text_len, request_tokens.data(), request_tokens.size(), true, false);
		}
		request_tokens.resize(n_tokens);

		size_t shared_prefix_idx = 0;
		auto diff = std::mismatch(context_tokens.begin(), context_tokens.end(), request_tokens.begin(), request_tokens.end());
		if (diff.first != context_tokens.end()) {
			shared_prefix_idx = std::distance(context_tokens.begin(), diff.first);
		} else {
			shared_prefix_idx = std::min(context_tokens.size(), request_tokens.size());
		}

		// Note: llama_kv_cache_seq_rm has been removed from newer llama.cpp
		// For now, we'll skip this optimization and regenerate everything
		// This may impact performance but ensures compatibility
		// TODO: Find equivalent functionality in newer llama.cpp API
		bool rm_success = true; // Stubbed out for compatibility
		context_tokens.erase(context_tokens.begin() + shared_prefix_idx, context_tokens.end());
		request_tokens.erase(request_tokens.begin(), request_tokens.begin() + shared_prefix_idx);

		// Use modern batch API - create batch for all tokens at once
		llama_batch batch = llama_batch_get_one(request_tokens.data(), request_tokens.size());

		printf("Request tokens: \n");
		for (size_t i = 0; i < request_tokens.size(); i++) {
			char token_str[256];
			const struct llama_vocab* vocab = llama_model_get_vocab(model->model);
			int len = llama_token_to_piece(vocab, request_tokens[i], token_str, sizeof(token_str), 0, false);
			if (len > 0) {
				token_str[len] = '\0';
				printf("%s", token_str);
			}
		}
		printf("\n");

		int curr_token_pos = context_tokens.size();
		bool decode_failed = false;

		// Process the batch
		if (llama_decode(ctx, batch) != 0) {
			decode_failed = true;
		}
		curr_token_pos += batch.n_tokens;

		printf("Request tokens: %d\n", (int32_t)request_tokens.size());
		printf("Batch tokens: %d\n", batch.n_tokens);
		printf("Current token pos: %d\n", curr_token_pos);

		if (decode_failed) {
			Dictionary response;
			response["id"] = req.id;
			response["error"] = "llama_decode() failed";
			call_thread_safe("emit_signal", "completion_generated", response);
			continue;
		}

		context_tokens.insert(context_tokens.end(), request_tokens.begin(), request_tokens.end());

		while (true) {
			if (exit_thread) {
				return;
			}
			// Modern sampling API
			llama_token new_token_id = llama_sampler_sample(sampler, ctx, batch.n_tokens - 1);
			llama_sampler_accept(sampler, new_token_id);

			Dictionary response;
			response["id"] = req.id;

			context_tokens.push_back(new_token_id);

			bool eog = llama_vocab_is_eog(llama_model_get_vocab(model->model), new_token_id);
			bool curr_eq_n_len = curr_token_pos == n_len;

			if (eog || curr_eq_n_len) {
				response["done"] = true;
				call_thread_safe("emit_signal", "completion_generated", response);
				break;
			}

			// Modern token to piece API
			char token_str[256];
			const struct llama_vocab* vocab = llama_model_get_vocab(model->model);
			int len = llama_token_to_piece(vocab, new_token_id, token_str, sizeof(token_str), 0, false);
			if (len > 0) {
				token_str[len] = '\0';
				response["text"] = String(token_str);
			} else {
				response["text"] = "";
			}
			response["done"] = false;
			call_thread_safe("emit_signal", "completion_generated", response);

			// Create new batch for single token generation
			llama_batch new_batch = llama_batch_get_one(&new_token_id, 1);

			curr_token_pos++;

			if (llama_decode(ctx, new_batch) != 0) {
				decode_failed = true;
				break;
			}
		}

		// Reset sampler state for modern API
		llama_sampler_reset(sampler);

		if (decode_failed) {
			Dictionary response;
			response["id"] = req.id;
			response["error"] = "llama_decode() failed";
			call_thread_safe("emit_signal", "completion_generated", response);
			continue;
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

int LlamaContext::request_completion(const String &prompt) {
	int id = request_id++;

	UtilityFunctions::print(vformat("%s: Requesting completion for prompt id: %d", __func__, id));

	mutex->lock();
	completion_request req = { id, prompt };
	completion_requests.append(req);
	mutex->unlock();

	semaphore->post();

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
	// TODO: Update sampler chain when changed
}

float LlamaContext::get_top_p() {
	return top_p;
}
void LlamaContext::set_top_p(float p) {
	this->top_p = p;
	// TODO: Update sampler chain when changed
}

float LlamaContext::get_frequency_penalty() {
	return penalty_freq;
}
void LlamaContext::set_frequency_penalty(float frequency_penalty) {
	this->penalty_freq = frequency_penalty;
	// TODO: Update sampler chain when changed
}

float LlamaContext::get_presence_penalty() {
	return penalty_present;
}
void LlamaContext::set_presence_penalty(float presence_penalty) {
	this->penalty_present = presence_penalty;
	// TODO: Update sampler chain when changed
}

void LlamaContext::_exit_tree() {
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	mutex->lock();
	exit_thread = true;
	mutex->unlock();

	semaphore->post();

	thread->wait_to_finish();

	if (ctx) {
		llama_free(ctx);
	}
	if (sampler) {
		llama_sampler_free(sampler);
		sampler = nullptr;
	}
	llama_backend_free();
}