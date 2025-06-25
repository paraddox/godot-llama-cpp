#include "batch_processor.h"
#include "common.h"
#include <godot_cpp/classes/time.hpp>
#include <algorithm>
#include <cstring>

using namespace godot;

BatchProcessor::BatchProcessor() {
    processor_mutex.instantiate();
}

BatchProcessor::~BatchProcessor() {
    shutdown();
}

bool BatchProcessor::initialize(Ref<LlamaModel> p_model, llama_context* p_ctx) {
    if (p_model.is_null() || p_model->model == nullptr || p_ctx == nullptr) {
        UtilityFunctions::printerr("BatchProcessor: Invalid model or context provided");
        return false;
    }
    
    model = p_model;
    ctx = p_ctx;
    
    // Create sampler for batch processing
    sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));
    
    initialize_token_pools();
    
    UtilityFunctions::print("BatchProcessor: Initialized successfully");
    return true;
}

void BatchProcessor::shutdown() {
    processor_mutex->lock();
    
    // Clean up active batches
    for (BatchGroup* batch : active_batches) {
        for (BatchRequest* req : batch->requests) {
            delete req;
        }
        delete batch;
    }
    active_batches.clear();
    
    // Clean up pending requests
    for (BatchRequest* req : pending_requests) {
        delete req;
    }
    pending_requests.clear();
    
    // Clean up completed requests
    for (BatchRequest* req : completed_requests) {
        delete req;
    }
    completed_requests.clear();
    
    processor_mutex->unlock();
    
    cleanup_token_pools();
    
    if (sampler) {
        llama_sampler_free(sampler);
        sampler = nullptr;
    }
}

int BatchProcessor::submit_request(const String& prompt, int32_t max_tokens) {
    static std::atomic<int> next_id{1};
    int request_id = next_id.fetch_add(1);
    
    BatchRequest* req = new BatchRequest();
    req->id = request_id;
    req->prompt = prompt;
    req->max_tokens = max_tokens;
    req->submit_time_ms = (uint64_t)Time::get_singleton()->get_time_dict_from_system()["unix"];
    
    // Tokenize prompt
    const char* text = prompt.utf8().get_data();
    int text_len = strlen(text);
    req->tokens.resize(text_len + 16);
    
    const struct llama_vocab* vocab = llama_model_get_vocab(model->model);
    int n_tokens = llama_tokenize(vocab, text, text_len, req->tokens.data(), req->tokens.size(), true, false);
    if (n_tokens < 0) {
        req->tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text, text_len, req->tokens.data(), req->tokens.size(), true, false);
    }
    req->tokens.resize(n_tokens);
    req->sequence_length = n_tokens;
    
    processor_mutex->lock();
    pending_requests.push_back(req);
    processor_mutex->unlock();
    
    return request_id;
}

void BatchProcessor::process_pending_requests() {
    processor_mutex->lock();
    
    if (pending_requests.empty()) {
        processor_mutex->unlock();
        return;
    }
    
    // Create new batch group
    BatchGroup* batch = create_batch_group();
    uint64_t current_time = (uint64_t)Time::get_singleton()->get_time_dict_from_system()["unix"];
    
    // Fill batch with pending requests
    while (!pending_requests.empty() && batch->requests.size() < max_batch_size) {
        BatchRequest* req = pending_requests.front();
        
        // Check if request fits in current batch
        if (batch->total_tokens + req->sequence_length > max_batch_size) {
            break;
        }
        
        pending_requests.pop_front();
        add_request_to_batch(req, batch);
    }
    
    processor_mutex->unlock();
    
    // Process batch if it meets criteria
    if (should_process_batch(batch)) {
        batch->processing = true;
        process_batch_group(batch);
        cleanup_completed_batch(batch);
    } else {
        // Return requests to pending queue if batch not ready
        processor_mutex->lock();
        for (auto it = batch->requests.rbegin(); it != batch->requests.rend(); ++it) {
            pending_requests.push_front(*it);
        }
        processor_mutex->unlock();
        delete batch;
    }
}

BatchGroup* BatchProcessor::create_batch_group() {
    BatchGroup* batch = new BatchGroup();
    batch->creation_time_ms = (uint64_t)Time::get_singleton()->get_time_dict_from_system()["unix"];
    return batch;
}

void BatchProcessor::add_request_to_batch(BatchRequest* req, BatchGroup* batch) {
    req->batch_position = batch->total_tokens;
    batch->requests.push_back(req);
    batch->total_tokens += req->sequence_length;
    batch->max_sequence_length = std::max(batch->max_sequence_length, req->sequence_length);
}

bool BatchProcessor::should_process_batch(BatchGroup* batch) {
    if (batch->requests.empty()) return false;
    
    uint64_t current_time = (uint64_t)Time::get_singleton()->get_time_dict_from_system()["unix"];
    uint64_t batch_age = current_time - batch->creation_time_ms;
    
    // Process if batch is full, timeout reached, or minimum size met
    return batch->requests.size() >= max_batch_size ||
           batch_age >= batch_timeout_ms ||
           (batch->requests.size() >= min_batch_size && pending_requests.empty());
}

void BatchProcessor::process_batch_group(BatchGroup* batch) {
    if (batch->requests.empty()) return;
    
    uint64_t start_time = (uint64_t)Time::get_singleton()->get_time_dict_from_system()["unix"];
    
    // Prepare consolidated batch for all requests
    std::vector<llama_token> batch_tokens;
    std::vector<size_t> request_positions;
    
    for (BatchRequest* req : batch->requests) {
        request_positions.push_back(batch_tokens.size());
        batch_tokens.insert(batch_tokens.end(), req->tokens.begin(), req->tokens.end());
    }
    
    // Process initial prompt batch
    if (!batch_tokens.empty()) {
        llama_batch llama_batch_data = llama_batch_get_one(batch_tokens.data(), batch_tokens.size());
        
        if (llama_decode(ctx, llama_batch_data) != 0) {
            UtilityFunctions::printerr("BatchProcessor: Failed to decode batch");
            for (BatchRequest* req : batch->requests) {
                req->failed = true;
                req->completed = true;
            }
            return;
        }
    }
    
    // Generate tokens for each request in parallel
    bool any_active = true;
    int generation_step = 0;
    
    while (any_active && generation_step < 512) { // Safety limit
        any_active = false;
        std::vector<llama_token> next_tokens;
        next_tokens.reserve(batch->requests.size());
        
        // Sample next token for each active request
        for (size_t i = 0; i < batch->requests.size(); i++) {
            BatchRequest* req = batch->requests[i];
            if (req->completed || req->failed) {
                next_tokens.push_back(0); // Placeholder
                continue;
            }
            
            // Sample token using current context position
            int token_pos = request_positions[i] + req->generated_tokens;
            llama_token new_token = llama_sampler_sample(sampler, ctx, token_pos);
            llama_sampler_accept(sampler, new_token);
            
            req->tokens.push_back(new_token);
            req->generated_tokens++;
            next_tokens.push_back(new_token);
            
            // Check completion conditions
            const struct llama_vocab* vocab = llama_model_get_vocab(model->model);
            bool eog = llama_vocab_is_eog(vocab, new_token);
            bool max_reached = req->generated_tokens >= req->max_tokens;
            
            if (eog || max_reached) {
                req->completed = true;
            } else {
                any_active = true;
            }
        }
        
        // Process next token batch if any requests are still active
        if (any_active) {
            llama_batch next_batch = llama_batch_get_one(next_tokens.data(), next_tokens.size());
            if (llama_decode(ctx, next_batch) != 0) {
                UtilityFunctions::printerr("BatchProcessor: Failed to decode generation step");
                break;
            }
        }
        
        generation_step++;
    }
    
    // Mark all requests as completed and move to completed queue
    processor_mutex->lock();
    for (BatchRequest* req : batch->requests) {
        if (!req->completed && !req->failed) {
            req->completed = true; // Timeout completion
        }
        completed_requests.push_back(req);
    }
    processor_mutex->unlock();
    
    // Update performance stats
    uint64_t current_time = (uint64_t)(uint64_t)Time::get_singleton()->get_time_dict_from_system()["unix"];
    uint64_t processing_time = current_time - start_time;
    total_requests_processed.fetch_add(batch->requests.size());
    total_processing_time_ms.fetch_add(processing_time);
    
    uint64_t tokens_generated = 0;
    for (BatchRequest* req : batch->requests) {
        tokens_generated += req->generated_tokens;
    }
    total_tokens_generated.fetch_add(tokens_generated);
    
    llama_sampler_reset(sampler);
}

void BatchProcessor::cleanup_completed_batch(BatchGroup* batch) {
    batch->requests.clear();
    delete batch;
}

std::vector<BatchRequest*> BatchProcessor::get_completed_requests() {
    processor_mutex->lock();
    std::vector<BatchRequest*> result = completed_requests;
    processor_mutex->unlock();
    return result;
}

void BatchProcessor::cleanup_completed_requests() {
    processor_mutex->lock();
    for (BatchRequest* req : completed_requests) {
        delete req;
    }
    completed_requests.clear();
    processor_mutex->unlock();
}

// Configuration methods
void BatchProcessor::set_batch_size(uint32_t min_size, uint32_t max_size) {
    min_batch_size = std::max(1u, min_size);
    max_batch_size = std::max(min_batch_size, max_size);
}

void BatchProcessor::set_batch_timeout(uint32_t timeout_ms) {
    batch_timeout_ms = timeout_ms;
}

void BatchProcessor::set_max_sequence_length(uint32_t length) {
    max_sequence_length = length;
}

// Memory management
void BatchProcessor::initialize_token_pools() {
    token_pools.resize(pool_size);
    pool_available.resize(pool_size, true);
    
    for (size_t i = 0; i < pool_size; i++) {
        token_pools[i] = new llama_token[pool_token_capacity];
    }
}

void BatchProcessor::cleanup_token_pools() {
    for (llama_token* pool : token_pools) {
        delete[] pool;
    }
    token_pools.clear();
    pool_available.clear();
}

llama_token* BatchProcessor::acquire_token_buffer() {
    for (size_t i = 0; i < pool_size; i++) {
        if (pool_available[i]) {
            pool_available[i] = false;
            return token_pools[i];
        }
    }
    
    // Fall back to dynamic allocation
    return new llama_token[pool_token_capacity];
}

void BatchProcessor::release_token_buffer(llama_token* buffer) {
    for (size_t i = 0; i < pool_size; i++) {
        if (token_pools[i] == buffer) {
            pool_available[i] = true;
            return;
        }
    }
    
    // Was dynamically allocated
    delete[] buffer;
}

BatchProcessor::BatchStats BatchProcessor::get_stats() const {
    BatchStats stats;
    stats.requests_processed = total_requests_processed.load();
    stats.tokens_generated = total_tokens_generated.load();
    stats.average_processing_time_ms = stats.requests_processed > 0 ? 
        total_processing_time_ms.load() / stats.requests_processed : 0;
    
    processor_mutex->lock();
    stats.active_batches = active_batches.size();
    stats.pending_requests = pending_requests.size();
    processor_mutex->unlock();
    
    stats.average_batch_utilization = stats.active_batches > 0 ? 
        (float)stats.pending_requests / (stats.active_batches * max_batch_size) * 100.0f : 0.0f;
    
    return stats;
}

void BatchProcessor::reset_stats() {
    total_requests_processed.store(0);
    total_tokens_generated.store(0);
    total_processing_time_ms.store(0);
}