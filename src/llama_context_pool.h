#ifndef LLAMA_CONTEXT_POOL_H
#define LLAMA_CONTEXT_POOL_H

#include "llama.h"
#include "llama_model.h"
#include "batch_processor.h"
#include "speculative_decoder.h"
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/semaphore.hpp>
#include <godot_cpp/classes/thread.hpp>
#include <godot_cpp/templates/vector.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <vector>
#include <atomic>

namespace godot {

struct PooledContext {
    llama_context* ctx = nullptr;
    struct llama_sampler* sampler = nullptr;
    BatchProcessor* batch_processor = nullptr;
    SpeculativeDecoder* speculative_decoder = nullptr;
    std::vector<llama_token> context_tokens;
    std::atomic<bool> busy{false};
    std::atomic<int> request_count{0};
    uint32_t context_id;
    bool healthy = true;
    bool batch_mode = true;
    bool speculative_mode = false;
};

struct ContextRequest {
    int id;
    String prompt;
    float temperature = 0.8f;
    float top_p = 0.95f;
    int32_t max_tokens = 1024;
    uint32_t preferred_context_id = UINT32_MAX; // Affinity hint
};

class LlamaContextPool {
private:
    Ref<LlamaModel> model;
    Vector<PooledContext*> contexts;
    Vector<ContextRequest> request_queue;
    
    Ref<Mutex> pool_mutex;
    Ref<Semaphore> request_semaphore;
    Vector<Ref<Thread>> worker_threads;
    Vector<uint32_t> thread_ids;
    
    std::atomic<bool> shutdown{false};
    std::atomic<int> next_request_id{1};
    
    uint32_t pool_size;
    llama_context_params base_params;
    bool batch_mode_enabled = true;
    bool speculative_mode_enabled = false;

    PooledContext* acquire_context();
    void release_context(PooledContext* ctx);
    void worker_thread_loop();
    void process_request(const ContextRequest& req, PooledContext* ctx);

public:
    LlamaContextPool();
    ~LlamaContextPool();
    
    bool initialize(Ref<LlamaModel> p_model, uint32_t p_pool_size = 1);  // Reduced from 4 to 1 for 4GB VRAM
    void shutdown_pool();
    
    int submit_request(const String& prompt, float temperature = 0.8f, 
                      float top_p = 0.95f, int32_t max_tokens = 1024,
                      uint32_t preferred_context = UINT32_MAX);
    
    // Batching controls
    void set_batch_mode(bool enabled);
    bool get_batch_mode() const;
    void set_batch_size(uint32_t min_size, uint32_t max_size);
    void set_batch_timeout(uint32_t timeout_ms);
    void process_batched_requests();
    
    // Speculative decoding controls
    void set_speculative_mode(bool enabled);
    bool get_speculative_mode() const;
    void set_lookahead_tokens(uint32_t tokens);
    uint32_t get_lookahead_tokens() const;
    void set_acceptance_threshold(float threshold);
    float get_acceptance_threshold() const;
    
    // Performance monitoring
    struct PoolStats {
        uint32_t active_contexts;
        uint32_t total_contexts;
        uint32_t queued_requests;
        uint64_t total_requests_processed;
        float average_gpu_utilization;
        uint64_t batched_requests_processed;
        float average_batch_efficiency;
    };
    
    PoolStats get_stats() const;
    void reset_unhealthy_contexts();
};

} // namespace godot

#endif