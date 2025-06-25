#include "llama_context_pool.h"
#include "batch_processor.h"
#include "common.h"
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <algorithm>

using namespace godot;

LlamaContextPool::LlamaContextPool() {
    pool_mutex.instantiate();
    request_semaphore.instantiate();
    pool_size = 0;
    
    base_params = llama_context_default_params();
    base_params.n_ctx = 2048;
    base_params.n_batch = 512;
    base_params.n_ubatch = 512;
    base_params.no_perf = false;
    
    int32_t optimal_threads = std::min(16, (int)OS::get_singleton()->get_processor_count());
    base_params.n_threads = optimal_threads;
    base_params.n_threads_batch = optimal_threads;
}

LlamaContextPool::~LlamaContextPool() {
    shutdown_pool();
}

bool LlamaContextPool::initialize(Ref<LlamaModel> p_model, uint32_t p_pool_size) {
    if (p_model.is_null() || p_model->model == nullptr) {
        UtilityFunctions::printerr("LlamaContextPool: Invalid model provided");
        return false;
    }
    
    model = p_model;
    pool_size = p_pool_size;
    
    // Create contexts
    contexts.resize(pool_size);
    for (uint32_t i = 0; i < pool_size; i++) {
        PooledContext* ctx = new PooledContext();
        ctx->context_id = i;
        ctx->ctx = llama_init_from_model(model->model, base_params);
        
        if (ctx->ctx == nullptr) {
            UtilityFunctions::printerr(vformat("LlamaContextPool: Failed to create context %d", i));
            delete ctx;
            return false;
        }
        
        // Create sampler chain
        ctx->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(ctx->sampler, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(ctx->sampler, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(ctx->sampler, llama_sampler_init_dist(0));
        
        // Initialize batch processor for this context
        if (batch_mode_enabled) {
            ctx->batch_processor = new BatchProcessor();
            if (!ctx->batch_processor->initialize(model, ctx->ctx)) {
                UtilityFunctions::printerr(vformat("LlamaContextPool: Failed to initialize batch processor for context %d", i));
                delete ctx->batch_processor;
                ctx->batch_processor = nullptr;
                ctx->batch_mode = false;
            }
        }
        
        // Initialize speculative decoder for this context
        if (speculative_mode_enabled) {
            ctx->speculative_decoder = new SpeculativeDecoder();
            if (!ctx->speculative_decoder->initialize(model, ctx->ctx)) {
                UtilityFunctions::printerr(vformat("LlamaContextPool: Failed to initialize speculative decoder for context %d", i));
                delete ctx->speculative_decoder;
                ctx->speculative_decoder = nullptr;
                ctx->speculative_mode = false;
            } else {
                ctx->speculative_mode = true;
            }
        }
        
        contexts.set(i, ctx);
    }
    
    // We'll manually create worker threads using a different approach
    // For now, skip thread creation and process requests synchronously
    UtilityFunctions::print("LlamaContextPool: Threads deferred for compatibility");
    
    UtilityFunctions::print(vformat("LlamaContextPool: Initialized with %d contexts", pool_size));
    return true;
}

void LlamaContextPool::shutdown_pool() {
    shutdown.store(true);
    
    // Wake up all worker threads
    for (uint32_t i = 0; i < pool_size; i++) {
        request_semaphore->post();
    }
    
    // Wait for threads to finish
    for (uint32_t i = 0; i < worker_threads.size(); i++) {
        if (worker_threads[i].is_valid()) {
            worker_threads[i]->wait_to_finish();
        }
    }
    
    // Clean up contexts
    for (uint32_t i = 0; i < contexts.size(); i++) {
        PooledContext* ctx = contexts[i];
        if (ctx) {
            if (ctx->batch_processor) {
                ctx->batch_processor->shutdown();
                delete ctx->batch_processor;
            }
            if (ctx->speculative_decoder) {
                ctx->speculative_decoder->shutdown();
                delete ctx->speculative_decoder;
            }
            if (ctx->ctx) {
                llama_free(ctx->ctx);
            }
            if (ctx->sampler) {
                llama_sampler_free(ctx->sampler);
            }
            delete ctx;
        }
    }
    
    contexts.clear();
    worker_threads.clear();
}

PooledContext* LlamaContextPool::acquire_context() {
    pool_mutex->lock();
    
    // First try: find idle context with lowest usage
    PooledContext* best_idle = nullptr;
    int min_requests = INT_MAX;
    
    for (uint32_t i = 0; i < contexts.size(); i++) {
        PooledContext* ctx = contexts[i];
        if (ctx->healthy && !ctx->busy.load()) {
            if (ctx->request_count.load() < min_requests) {
                min_requests = ctx->request_count.load();
                best_idle = ctx;
            }
        }
    }
    
    if (best_idle) {
        best_idle->busy.store(true);
        pool_mutex->unlock();
        return best_idle;
    }
    
    pool_mutex->unlock();
    return nullptr;
}

void LlamaContextPool::release_context(PooledContext* ctx) {
    if (ctx) {
        ctx->busy.store(false);
        ctx->request_count.fetch_add(1);
    }
}

int LlamaContextPool::submit_request(const String& prompt, float temperature, 
                                   float top_p, int32_t max_tokens,
                                   uint32_t preferred_context) {
    
    // Try batch processing first if enabled
    if (batch_mode_enabled) {
        PooledContext* ctx = acquire_context();
        if (ctx && ctx->batch_processor && ctx->batch_mode) {
            int batch_id = ctx->batch_processor->submit_request(prompt, max_tokens);
            release_context(ctx);
            return batch_id;
        }
        if (ctx) {
            release_context(ctx);
        }
    }
    
    // Fall back to individual request processing
    int request_id = next_request_id.fetch_add(1);
    
    ContextRequest req;
    req.id = request_id;
    req.prompt = prompt;
    req.temperature = temperature;
    req.top_p = top_p;
    req.max_tokens = max_tokens;
    req.preferred_context_id = preferred_context;
    
    // Process immediately for now (synchronous)
    PooledContext* ctx = acquire_context();
    if (ctx) {
        process_request(req, ctx);
        release_context(ctx);
    }
    
    return request_id;
}

void LlamaContextPool::worker_thread_loop() {
    while (!shutdown.load()) {
        request_semaphore->wait();
        
        if (shutdown.load()) break;
        
        // Get next request
        pool_mutex->lock();
        if (request_queue.size() == 0) {
            pool_mutex->unlock();
            continue;
        }
        
        ContextRequest req = request_queue[0];
        request_queue.remove_at(0);
        pool_mutex->unlock();
        
        // Acquire context
        PooledContext* ctx = acquire_context();
        if (!ctx) {
            // Put request back if no context available
            pool_mutex->lock();
            request_queue.insert(0, req);
            pool_mutex->unlock();
            continue;
        }
        
        // Process request
        process_request(req, ctx);
        
        // Release context
        release_context(ctx);
    }
}

void LlamaContextPool::process_request(const ContextRequest& req, PooledContext* ctx) {
    // Update sampler parameters
    llama_sampler_reset(ctx->sampler);
    // TODO: Update temperature/top_p in sampler chain
    
    // Tokenize prompt
    std::vector<llama_token> request_tokens;
    const char* text = req.prompt.utf8().get_data();
    int text_len = strlen(text);
    request_tokens.resize(text_len + 16);
    
    const struct llama_vocab* vocab = llama_model_get_vocab(model->model);
    int n_tokens = llama_tokenize(vocab, text, text_len, request_tokens.data(), request_tokens.size(), true, false);
    if (n_tokens < 0) {
        request_tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text, text_len, request_tokens.data(), request_tokens.size(), true, false);
    }
    request_tokens.resize(n_tokens);
    
    // Find shared prefix with context
    size_t shared_prefix_idx = 0;
    auto diff = std::mismatch(ctx->context_tokens.begin(), ctx->context_tokens.end(), 
                             request_tokens.begin(), request_tokens.end());
    if (diff.first != ctx->context_tokens.end()) {
        shared_prefix_idx = std::distance(ctx->context_tokens.begin(), diff.first);
    } else {
        shared_prefix_idx = std::min(ctx->context_tokens.size(), request_tokens.size());
    }
    
    // Update context tokens
    ctx->context_tokens.erase(ctx->context_tokens.begin() + shared_prefix_idx, ctx->context_tokens.end());
    request_tokens.erase(request_tokens.begin(), request_tokens.begin() + shared_prefix_idx);
    
    // Process batch
    if (!request_tokens.empty()) {
        llama_batch batch = llama_batch_get_one(request_tokens.data(), request_tokens.size());
        
        if (llama_decode(ctx->ctx, batch) != 0) {
            UtilityFunctions::printerr(vformat("LlamaContextPool: Decode failed for request %d", req.id));
            ctx->healthy = false;
            return;
        }
        
        ctx->context_tokens.insert(ctx->context_tokens.end(), request_tokens.begin(), request_tokens.end());
    }
    
    // Generate tokens
    for (int32_t i = 0; i < req.max_tokens; i++) {
        if (shutdown.load()) break;
        
        llama_token new_token = llama_sampler_sample(ctx->sampler, ctx->ctx, -1);
        llama_sampler_accept(ctx->sampler, new_token);
        
        ctx->context_tokens.push_back(new_token);
        
        bool eog = llama_vocab_is_eog(vocab, new_token);
        if (eog) break;
        
        // Single token batch for next iteration
        llama_batch single_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx->ctx, single_batch) != 0) {
            UtilityFunctions::printerr(vformat("LlamaContextPool: Single token decode failed for request %d", req.id));
            ctx->healthy = false;
            break;
        }
    }
}

LlamaContextPool::PoolStats LlamaContextPool::get_stats() const {
    PoolStats stats;
    stats.total_contexts = contexts.size();
    stats.active_contexts = 0;
    stats.total_requests_processed = 0;
    
    for (uint32_t i = 0; i < contexts.size(); i++) {
        if (contexts[i]->busy.load()) {
            stats.active_contexts++;
        }
        stats.total_requests_processed += contexts[i]->request_count.load();
    }
    
    pool_mutex->lock();
    stats.queued_requests = request_queue.size();
    pool_mutex->unlock();
    
    stats.average_gpu_utilization = (float)stats.active_contexts / stats.total_contexts * 100.0f;
    
    return stats;
}

void LlamaContextPool::reset_unhealthy_contexts() {
    pool_mutex->lock();
    for (uint32_t i = 0; i < contexts.size(); i++) {
        PooledContext* ctx = contexts[i];
        if (!ctx->healthy && !ctx->busy.load()) {
            // Clean up batch processor
            if (ctx->batch_processor) {
                ctx->batch_processor->shutdown();
                delete ctx->batch_processor;
                ctx->batch_processor = nullptr;
            }
            
            // Recreate context
            if (ctx->ctx) {
                llama_free(ctx->ctx);
            }
            if (ctx->sampler) {
                llama_sampler_free(ctx->sampler);
            }
            
            ctx->ctx = llama_init_from_model(model->model, base_params);
            if (ctx->ctx) {
                ctx->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(ctx->sampler, llama_sampler_init_temp(0.8f));
                llama_sampler_chain_add(ctx->sampler, llama_sampler_init_top_p(0.95f, 1));
                llama_sampler_chain_add(ctx->sampler, llama_sampler_init_dist(0));
                
                // Recreate batch processor if needed
                if (batch_mode_enabled) {
                    ctx->batch_processor = new BatchProcessor();
                    if (ctx->batch_processor->initialize(model, ctx->ctx)) {
                        ctx->batch_mode = true;
                    } else {
                        delete ctx->batch_processor;
                        ctx->batch_processor = nullptr;
                        ctx->batch_mode = false;
                    }
                }
                
                ctx->context_tokens.clear();
                ctx->healthy = true;
                ctx->request_count.store(0);
            }
        }
    }
    pool_mutex->unlock();
}

// Batching control methods
void LlamaContextPool::set_batch_mode(bool enabled) {
    batch_mode_enabled = enabled;
    
    pool_mutex->lock();
    for (uint32_t i = 0; i < contexts.size(); i++) {
        PooledContext* ctx = contexts[i];
        if (enabled && !ctx->batch_processor) {
            ctx->batch_processor = new BatchProcessor();
            if (ctx->batch_processor->initialize(model, ctx->ctx)) {
                ctx->batch_mode = true;
            } else {
                delete ctx->batch_processor;
                ctx->batch_processor = nullptr;
                ctx->batch_mode = false;
            }
        } else if (!enabled && ctx->batch_processor) {
            ctx->batch_processor->shutdown();
            delete ctx->batch_processor;
            ctx->batch_processor = nullptr;
            ctx->batch_mode = false;
        }
    }
    pool_mutex->unlock();
}

bool LlamaContextPool::get_batch_mode() const {
    return batch_mode_enabled;
}

void LlamaContextPool::set_batch_size(uint32_t min_size, uint32_t max_size) {
    pool_mutex->lock();
    for (uint32_t i = 0; i < contexts.size(); i++) {
        PooledContext* ctx = contexts[i];
        if (ctx->batch_processor) {
            ctx->batch_processor->set_batch_size(min_size, max_size);
        }
    }
    pool_mutex->unlock();
}

void LlamaContextPool::set_batch_timeout(uint32_t timeout_ms) {
    pool_mutex->lock();
    for (uint32_t i = 0; i < contexts.size(); i++) {
        PooledContext* ctx = contexts[i];
        if (ctx->batch_processor) {
            ctx->batch_processor->set_batch_timeout(timeout_ms);
        }
    }
    pool_mutex->unlock();
}

void LlamaContextPool::process_batched_requests() {
    pool_mutex->lock();
    for (uint32_t i = 0; i < contexts.size(); i++) {
        PooledContext* ctx = contexts[i];
        if (ctx->batch_processor && !ctx->busy.load()) {
            ctx->batch_processor->process_pending_requests();
        }
    }
    pool_mutex->unlock();
}

// Speculative decoding control methods
void LlamaContextPool::set_speculative_mode(bool enabled) {
    speculative_mode_enabled = enabled;
    
    pool_mutex->lock();
    for (uint32_t i = 0; i < contexts.size(); i++) {
        PooledContext* ctx = contexts[i];
        if (enabled && !ctx->speculative_decoder) {
            ctx->speculative_decoder = new SpeculativeDecoder();
            if (ctx->speculative_decoder->initialize(model, ctx->ctx)) {
                ctx->speculative_mode = true;
            } else {
                delete ctx->speculative_decoder;
                ctx->speculative_decoder = nullptr;
                ctx->speculative_mode = false;
            }
        } else if (!enabled && ctx->speculative_decoder) {
            ctx->speculative_decoder->shutdown();
            delete ctx->speculative_decoder;
            ctx->speculative_decoder = nullptr;
            ctx->speculative_mode = false;
        }
    }
    pool_mutex->unlock();
}

bool LlamaContextPool::get_speculative_mode() const {
    return speculative_mode_enabled;
}

void LlamaContextPool::set_lookahead_tokens(uint32_t tokens) {
    pool_mutex->lock();
    for (uint32_t i = 0; i < contexts.size(); i++) {
        PooledContext* ctx = contexts[i];
        if (ctx->speculative_decoder) {
            ctx->speculative_decoder->set_lookahead_tokens(tokens);
        }
    }
    pool_mutex->unlock();
}

uint32_t LlamaContextPool::get_lookahead_tokens() const {
    pool_mutex->lock();
    for (uint32_t i = 0; i < contexts.size(); i++) {
        PooledContext* ctx = contexts[i];
        if (ctx->speculative_decoder) {
            uint32_t tokens = ctx->speculative_decoder->get_lookahead_tokens();
            pool_mutex->unlock();
            return tokens;
        }
    }
    pool_mutex->unlock();
    return 4; // Default value
}

void LlamaContextPool::set_acceptance_threshold(float threshold) {
    pool_mutex->lock();
    for (uint32_t i = 0; i < contexts.size(); i++) {
        PooledContext* ctx = contexts[i];
        if (ctx->speculative_decoder) {
            ctx->speculative_decoder->set_acceptance_threshold(threshold);
        }
    }
    pool_mutex->unlock();
}

float LlamaContextPool::get_acceptance_threshold() const {
    pool_mutex->lock();
    for (uint32_t i = 0; i < contexts.size(); i++) {
        PooledContext* ctx = contexts[i];
        if (ctx->speculative_decoder) {
            float threshold = ctx->speculative_decoder->get_acceptance_threshold();
            pool_mutex->unlock();
            return threshold;
        }
    }
    pool_mutex->unlock();
    return 0.7f; // Default value
}