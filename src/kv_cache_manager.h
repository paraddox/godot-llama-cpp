#ifndef KV_CACHE_MANAGER_H
#define KV_CACHE_MANAGER_H

#include "llama.h"
#include "llama_model.h"
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <vector>
#include <unordered_map>
#include <atomic>

namespace godot {

struct CacheEntry {
    std::vector<llama_token> tokens;
    size_t sequence_id;
    size_t start_position;
    size_t length;
    uint64_t last_access_time;
    bool active = true;
};

struct CacheStats {
    size_t total_sequences;
    size_t active_sequences;
    size_t cache_hits;
    size_t cache_misses;
    float hit_ratio;
    size_t memory_usage_bytes;
    size_t max_memory_bytes;
};

class KVCacheManager {
private:
    llama_context* ctx;
    Ref<LlamaModel> model;
    
    // Cache management
    std::unordered_map<size_t, CacheEntry> cache_entries;
    std::vector<size_t> free_sequence_ids;
    size_t next_sequence_id = 1;
    
    // Memory management
    size_t max_cache_size_mb = 1024;  // 1GB default
    size_t current_memory_usage = 0;
    size_t context_size;
    size_t token_size_bytes = 4;  // 32-bit tokens
    
    // Performance tracking
    std::atomic<size_t> cache_hits{0};
    std::atomic<size_t> cache_misses{0};
    std::atomic<size_t> evictions{0};
    
    Ref<Mutex> cache_mutex;
    
    // Core cache operations
    size_t allocate_sequence_id();
    void deallocate_sequence_id(size_t seq_id);
    bool has_cache_space(size_t required_tokens);
    void evict_oldest_sequences(size_t required_space);
    size_t calculate_sequence_memory_usage(size_t length);
    
    // Cache optimization
    void defragment_cache();
    void optimize_cache_layout();
    std::vector<size_t> find_mergeable_sequences();
    
    // Sequence management using modern llama.cpp KV cache API
    bool clear_sequence_range(size_t seq_id, size_t start_pos, size_t end_pos);
    bool copy_sequence_range(size_t src_seq_id, size_t dst_seq_id, 
                           size_t src_start, size_t dst_start, size_t length);

public:
    KVCacheManager();
    ~KVCacheManager();
    
    bool initialize(llama_context* p_ctx, Ref<LlamaModel> p_model);
    void shutdown();
    
    // Sequence management
    size_t create_sequence(const std::vector<llama_token>& initial_tokens);
    bool append_to_sequence(size_t seq_id, const std::vector<llama_token>& tokens);
    bool truncate_sequence(size_t seq_id, size_t new_length);
    bool remove_sequence(size_t seq_id);
    
    // Cache operations
    bool find_common_prefix(const std::vector<llama_token>& tokens, 
                          size_t& best_seq_id, size_t& common_length);
    size_t fork_sequence(size_t parent_seq_id, size_t fork_position);
    bool merge_sequences(size_t seq_id1, size_t seq_id2);
    
    // Memory management
    void set_max_cache_size_mb(size_t size_mb);
    size_t get_max_cache_size_mb() const;
    void cleanup_inactive_sequences();
    void force_garbage_collection();
    
    // Performance optimization
    void preload_common_prefixes(const std::vector<std::vector<llama_token>>& common_sequences);
    void optimize_for_batch_processing();
    void update_access_time(size_t seq_id);
    
    // Monitoring and debugging
    CacheStats get_stats() const;
    void reset_stats();
    std::vector<size_t> get_active_sequences() const;
    size_t get_sequence_length(size_t seq_id) const;
    
    // Configuration
    void set_context_size(size_t size);
    size_t get_context_size() const;
    bool is_sequence_active(size_t seq_id) const;
};

} // namespace godot

#endif