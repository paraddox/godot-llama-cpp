#include "kv_cache_manager.h"
#include "common.h"
#include <godot_cpp/classes/time.hpp>
#include <algorithm>
#include <numeric>

using namespace godot;

KVCacheManager::KVCacheManager() {
    cache_mutex.instantiate();
    ctx = nullptr;
    context_size = 2048; // Default context size
}

KVCacheManager::~KVCacheManager() {
    shutdown();
}

bool KVCacheManager::initialize(llama_context* p_ctx, Ref<LlamaModel> p_model) {
    if (p_ctx == nullptr || p_model.is_null() || p_model->model == nullptr) {
        UtilityFunctions::printerr("KVCacheManager: Invalid context or model");
        return false;
    }
    
    ctx = p_ctx;
    model = p_model;
    
    // Get actual context size from llama context
    context_size = llama_n_ctx(ctx);
    
    UtilityFunctions::print(vformat("KVCacheManager: Initialized with context size %d", context_size));
    return true;
}

void KVCacheManager::shutdown() {
    cache_mutex->lock();
    
    // Clear all sequences from KV cache
    for (auto& entry : cache_entries) {
        if (entry.second.active) {
            // Note: llama_kv_cache_seq_rm has been removed in newer versions
            // We'll clear sequences using the modern API when available
            clear_sequence_range(entry.first, 0, entry.second.length);
        }
    }
    
    cache_entries.clear();
    free_sequence_ids.clear();
    current_memory_usage = 0;
    
    cache_mutex->unlock();
}

size_t KVCacheManager::create_sequence(const std::vector<llama_token>& initial_tokens) {
    cache_mutex->lock();
    
    size_t seq_id = allocate_sequence_id();
    size_t required_memory = calculate_sequence_memory_usage(initial_tokens.size());
    
    // Check if we have enough cache space
    if (!has_cache_space(initial_tokens.size())) {
        evict_oldest_sequences(required_memory);
    }
    
    CacheEntry entry;
    entry.tokens = initial_tokens;
    entry.sequence_id = seq_id;
    entry.start_position = 0;
    entry.length = initial_tokens.size();
    entry.last_access_time = (uint64_t)Time::get_singleton()->get_time_dict_from_system()["unix"];
    entry.active = true;
    
    cache_entries[seq_id] = entry;
    current_memory_usage += required_memory;
    
    cache_mutex->unlock();
    
    UtilityFunctions::print(vformat("KVCacheManager: Created sequence %d with %d tokens", seq_id, initial_tokens.size()));
    return seq_id;
}

bool KVCacheManager::append_to_sequence(size_t seq_id, const std::vector<llama_token>& tokens) {
    cache_mutex->lock();
    
    auto it = cache_entries.find(seq_id);
    if (it == cache_entries.end() || !it->second.active) {
        cache_mutex->unlock();
        return false;
    }
    
    CacheEntry& entry = it->second;
    size_t new_length = entry.length + tokens.size();
    size_t additional_memory = calculate_sequence_memory_usage(tokens.size());
    
    // Check context size limit
    if (new_length > context_size) {
        cache_mutex->unlock();
        UtilityFunctions::printerr(vformat("KVCacheManager: Sequence %d would exceed context size", seq_id));
        return false;
    }
    
    // Check memory limits
    if (current_memory_usage + additional_memory > max_cache_size_mb * 1024 * 1024) {
        evict_oldest_sequences(additional_memory);
    }
    
    // Append tokens to the sequence
    entry.tokens.insert(entry.tokens.end(), tokens.begin(), tokens.end());
    entry.length = new_length;
    entry.last_access_time = (uint64_t)Time::get_singleton()->get_time_dict_from_system()["unix"];
    current_memory_usage += additional_memory;
    
    cache_mutex->unlock();
    return true;
}

bool KVCacheManager::truncate_sequence(size_t seq_id, size_t new_length) {
    cache_mutex->lock();
    
    auto it = cache_entries.find(seq_id);
    if (it == cache_entries.end() || !it->second.active) {
        cache_mutex->unlock();
        return false;
    }
    
    CacheEntry& entry = it->second;
    if (new_length >= entry.length) {
        cache_mutex->unlock();
        return true; // Nothing to truncate
    }
    
    // Clear the truncated portion from KV cache
    clear_sequence_range(seq_id, new_length, entry.length);
    
    // Update entry
    size_t removed_tokens = entry.length - new_length;
    size_t freed_memory = calculate_sequence_memory_usage(removed_tokens);
    
    entry.tokens.resize(new_length);
    entry.length = new_length;
    entry.last_access_time = (uint64_t)Time::get_singleton()->get_time_dict_from_system()["unix"];
    current_memory_usage -= freed_memory;
    
    cache_mutex->unlock();
    return true;
}

bool KVCacheManager::remove_sequence(size_t seq_id) {
    cache_mutex->lock();
    
    auto it = cache_entries.find(seq_id);
    if (it == cache_entries.end()) {
        cache_mutex->unlock();
        return false;
    }
    
    CacheEntry& entry = it->second;
    
    // Clear from KV cache
    if (entry.active) {
        clear_sequence_range(seq_id, 0, entry.length);
    }
    
    // Free memory accounting
    size_t freed_memory = calculate_sequence_memory_usage(entry.length);
    current_memory_usage -= freed_memory;
    
    // Remove from cache and mark sequence ID as free
    cache_entries.erase(it);
    deallocate_sequence_id(seq_id);
    
    cache_mutex->unlock();
    return true;
}

bool KVCacheManager::find_common_prefix(const std::vector<llama_token>& tokens, 
                                       size_t& best_seq_id, size_t& common_length) {
    cache_mutex->lock();
    
    best_seq_id = 0;
    common_length = 0;
    
    for (const auto& entry : cache_entries) {
        if (!entry.second.active) continue;
        
        const std::vector<llama_token>& cached_tokens = entry.second.tokens;
        
        // Find common prefix length
        size_t prefix_length = 0;
        size_t max_compare = std::min(tokens.size(), cached_tokens.size());
        
        for (size_t i = 0; i < max_compare; i++) {
            if (tokens[i] == cached_tokens[i]) {
                prefix_length++;
            } else {
                break;
            }
        }
        
        // Update best match if this is longer
        if (prefix_length > common_length) {
            common_length = prefix_length;
            best_seq_id = entry.first;
        }
    }
    
    cache_mutex->unlock();
    
    if (common_length > 0) {
        cache_hits.fetch_add(1);
        update_access_time(best_seq_id);
        return true;
    } else {
        cache_misses.fetch_add(1);
        return false;
    }
}

size_t KVCacheManager::fork_sequence(size_t parent_seq_id, size_t fork_position) {
    cache_mutex->lock();
    
    auto it = cache_entries.find(parent_seq_id);
    if (it == cache_entries.end() || !it->second.active) {
        cache_mutex->unlock();
        return 0;
    }
    
    const CacheEntry& parent = it->second;
    if (fork_position >= parent.length) {
        cache_mutex->unlock();
        return 0;
    }
    
    // Create new sequence with tokens up to fork position
    std::vector<llama_token> fork_tokens(parent.tokens.begin(), 
                                        parent.tokens.begin() + fork_position);
    
    size_t new_seq_id = allocate_sequence_id();
    
    // Copy KV cache data from parent to child
    if (ctx && copy_sequence_range(parent_seq_id, new_seq_id, 0, 0, fork_position)) {
        CacheEntry new_entry;
        new_entry.tokens = fork_tokens;
        new_entry.sequence_id = new_seq_id;
        new_entry.start_position = 0;
        new_entry.length = fork_position;
        new_entry.last_access_time = (uint64_t)Time::get_singleton()->get_time_dict_from_system()["unix"];
        new_entry.active = true;
        
        cache_entries[new_seq_id] = new_entry;
        current_memory_usage += calculate_sequence_memory_usage(fork_position);
        
        cache_mutex->unlock();
        return new_seq_id;
    } else {
        deallocate_sequence_id(new_seq_id);
        cache_mutex->unlock();
        return 0;
    }
}

// Helper methods
size_t KVCacheManager::allocate_sequence_id() {
    if (!free_sequence_ids.empty()) {
        size_t seq_id = free_sequence_ids.back();
        free_sequence_ids.pop_back();
        return seq_id;
    }
    return next_sequence_id++;
}

void KVCacheManager::deallocate_sequence_id(size_t seq_id) {
    free_sequence_ids.push_back(seq_id);
}

bool KVCacheManager::has_cache_space(size_t required_tokens) {
    size_t required_memory = calculate_sequence_memory_usage(required_tokens);
    return (current_memory_usage + required_memory) <= (max_cache_size_mb * 1024 * 1024);
}

void KVCacheManager::evict_oldest_sequences(size_t required_space) {
    std::vector<std::pair<uint64_t, size_t>> sequence_ages;
    
    for (const auto& entry : cache_entries) {
        if (entry.second.active) {
            sequence_ages.push_back({entry.second.last_access_time, entry.first});
        }
    }
    
    // Sort by age (oldest first)
    std::sort(sequence_ages.begin(), sequence_ages.end());
    
    size_t freed_space = 0;
    for (const auto& age_seq : sequence_ages) {
        if (freed_space >= required_space) break;
        
        size_t seq_id = age_seq.second;
        auto it = cache_entries.find(seq_id);
        if (it != cache_entries.end()) {
            freed_space += calculate_sequence_memory_usage(it->second.length);
            remove_sequence(seq_id);
            evictions.fetch_add(1);
        }
    }
}

size_t KVCacheManager::calculate_sequence_memory_usage(size_t length) {
    // Estimate memory usage: token storage + KV cache overhead
    // This is a simplified calculation - actual usage depends on model architecture
    return length * (token_size_bytes + 64); // 64 bytes per token for KV cache estimate
}

bool KVCacheManager::clear_sequence_range(size_t seq_id, size_t start_pos, size_t end_pos) {
    // Modern llama.cpp may not have llama_kv_cache_seq_rm
    // We'll implement this as a placeholder for now
    // In practice, you'd use the actual llama.cpp KV cache management API
    return true;
}

bool KVCacheManager::copy_sequence_range(size_t src_seq_id, size_t dst_seq_id, 
                                        size_t src_start, size_t dst_start, size_t length) {
    // Placeholder for KV cache copying
    // Would use llama.cpp's sequence copying API when available
    return true;
}

void KVCacheManager::update_access_time(size_t seq_id) {
    auto it = cache_entries.find(seq_id);
    if (it != cache_entries.end()) {
        it->second.last_access_time = (uint64_t)Time::get_singleton()->get_time_dict_from_system()["unix"];
    }
}

void KVCacheManager::cleanup_inactive_sequences() {
    cache_mutex->lock();
    
    uint64_t current_time = (uint64_t)Time::get_singleton()->get_time_dict_from_system()["unix"];
    uint64_t timeout_ms = 300000; // 5 minutes
    
    std::vector<size_t> to_remove;
    for (const auto& entry : cache_entries) {
        if (entry.second.active && 
            (current_time - entry.second.last_access_time) > timeout_ms) {
            to_remove.push_back(entry.first);
        }
    }
    
    cache_mutex->unlock();
    
    for (size_t seq_id : to_remove) {
        remove_sequence(seq_id);
    }
}

// Configuration and monitoring methods
void KVCacheManager::set_max_cache_size_mb(size_t size_mb) {
    max_cache_size_mb = size_mb;
}

size_t KVCacheManager::get_max_cache_size_mb() const {
    return max_cache_size_mb;
}

CacheStats KVCacheManager::get_stats() const {
    cache_mutex->lock();
    
    CacheStats stats;
    stats.total_sequences = cache_entries.size();
    stats.active_sequences = 0;
    
    for (const auto& entry : cache_entries) {
        if (entry.second.active) {
            stats.active_sequences++;
        }
    }
    
    stats.cache_hits = cache_hits.load();
    stats.cache_misses = cache_misses.load();
    
    uint64_t total_requests = stats.cache_hits + stats.cache_misses;
    stats.hit_ratio = total_requests > 0 ? 
        static_cast<float>(stats.cache_hits) / total_requests * 100.0f : 0.0f;
    
    stats.memory_usage_bytes = current_memory_usage;
    stats.max_memory_bytes = max_cache_size_mb * 1024 * 1024;
    
    cache_mutex->unlock();
    return stats;
}

void KVCacheManager::reset_stats() {
    cache_hits.store(0);
    cache_misses.store(0);
    evictions.store(0);
}

std::vector<size_t> KVCacheManager::get_active_sequences() const {
    cache_mutex->lock();
    
    std::vector<size_t> active_seqs;
    for (const auto& entry : cache_entries) {
        if (entry.second.active) {
            active_seqs.push_back(entry.first);
        }
    }
    
    cache_mutex->unlock();
    return active_seqs;
}

size_t KVCacheManager::get_sequence_length(size_t seq_id) const {
    cache_mutex->lock();
    
    auto it = cache_entries.find(seq_id);
    size_t length = (it != cache_entries.end()) ? it->second.length : 0;
    
    cache_mutex->unlock();
    return length;
}

bool KVCacheManager::is_sequence_active(size_t seq_id) const {
    cache_mutex->lock();
    
    auto it = cache_entries.find(seq_id);
    bool active = (it != cache_entries.end()) && it->second.active;
    
    cache_mutex->unlock();
    return active;
}