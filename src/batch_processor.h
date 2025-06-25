#ifndef BATCH_PROCESSOR_H
#define BATCH_PROCESSOR_H

#include "llama.h"
#include "llama_model.h"
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <vector>
#include <deque>
#include <atomic>

namespace godot {

struct BatchRequest {
    int id;
    String prompt;
    std::vector<llama_token> tokens;
    int32_t max_tokens;
    int32_t generated_tokens = 0;
    bool completed = false;
    bool failed = false;
    
    // Batching metadata
    size_t batch_position = 0;
    size_t sequence_length = 0;
    uint64_t submit_time_ms = 0;
};

struct BatchGroup {
    std::vector<BatchRequest*> requests;
    size_t total_tokens = 0;
    size_t max_sequence_length = 0;
    uint64_t creation_time_ms = 0;
    bool processing = false;
};

class BatchProcessor {
private:
    Ref<LlamaModel> model;
    llama_context* ctx;
    struct llama_sampler* sampler;
    
    // Request management
    std::deque<BatchRequest*> pending_requests;
    std::vector<BatchGroup*> active_batches;
    std::vector<BatchRequest*> completed_requests;
    
    Ref<Mutex> processor_mutex;
    
    // Batching parameters
    uint32_t max_batch_size = 512;
    uint32_t min_batch_size = 1;
    uint32_t batch_timeout_ms = 50;  // Max wait time to fill batch
    uint32_t max_sequence_length = 2048;
    
    // Performance tracking
    std::atomic<uint64_t> total_requests_processed{0};
    std::atomic<uint64_t> total_tokens_generated{0};
    std::atomic<uint64_t> total_processing_time_ms{0};
    
    // Memory pools
    std::vector<llama_token*> token_pools;
    std::vector<bool> pool_available;
    size_t pool_size = 16;
    size_t pool_token_capacity = 4096;

    BatchGroup* create_batch_group();
    void add_request_to_batch(BatchRequest* req, BatchGroup* batch);
    bool should_process_batch(BatchGroup* batch);
    void process_batch_group(BatchGroup* batch);
    void cleanup_completed_batch(BatchGroup* batch);
    
    // Memory management
    llama_token* acquire_token_buffer();
    void release_token_buffer(llama_token* buffer);
    void initialize_token_pools();
    void cleanup_token_pools();

public:
    BatchProcessor();
    ~BatchProcessor();
    
    bool initialize(Ref<LlamaModel> p_model, llama_context* p_ctx);
    void shutdown();
    
    int submit_request(const String& prompt, int32_t max_tokens = 1024);
    void process_pending_requests();
    std::vector<BatchRequest*> get_completed_requests();
    void cleanup_completed_requests();
    
    // Configuration
    void set_batch_size(uint32_t min_size, uint32_t max_size);
    void set_batch_timeout(uint32_t timeout_ms);
    void set_max_sequence_length(uint32_t length);
    
    // Performance monitoring
    struct BatchStats {
        uint64_t requests_processed;
        uint64_t tokens_generated;
        uint64_t average_processing_time_ms;
        float average_batch_utilization;
        uint32_t active_batches;
        uint32_t pending_requests;
    };
    
    BatchStats get_stats() const;
    void reset_stats();
};

} // namespace godot

#endif