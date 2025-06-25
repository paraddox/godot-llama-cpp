#ifndef SPECULATIVE_DECODER_H
#define SPECULATIVE_DECODER_H

#include "llama.h"
#include "llama_model.h"
#include <godot_cpp/classes/mutex.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <vector>
#include <deque>
#include <atomic>

namespace godot {

struct SpeculativeCandidate {
    llama_token token;
    float probability;
    int depth; // How many tokens ahead this candidate is
};

struct SpeculativeSequence {
    std::vector<llama_token> draft_tokens;
    std::vector<float> draft_probabilities;
    std::vector<llama_token> verified_tokens;
    int sequence_id;
    size_t original_length;
    bool completed = false;
};

class SpeculativeDecoder {
private:
    // Model references
    Ref<LlamaModel> target_model;     // Main high-quality model
    Ref<LlamaModel> draft_model;      // Fast draft model (optional - can use same model)
    
    // Context references
    llama_context* target_ctx;
    llama_context* draft_ctx;
    
    // Samplers
    struct llama_sampler* target_sampler;
    struct llama_sampler* draft_sampler;
    
    // Speculative parameters
    uint32_t lookahead_tokens = 4;     // How many tokens to generate speculatively
    float acceptance_threshold = 0.7f;  // Probability threshold for accepting draft tokens
    uint32_t max_draft_length = 8;     // Maximum speculative sequence length
    bool use_separate_draft_model = false;
    
    // Performance tracking
    std::atomic<uint64_t> total_tokens_generated{0};
    std::atomic<uint64_t> accepted_draft_tokens{0};
    std::atomic<uint64_t> rejected_draft_tokens{0};
    std::atomic<uint64_t> speculation_cycles{0};
    
    // Working state
    std::vector<llama_token> context_tokens;
    std::deque<SpeculativeSequence> active_sequences;
    
    Ref<Mutex> decoder_mutex;

    // Core speculative decoding methods
    std::vector<llama_token> generate_draft_sequence(const std::vector<llama_token>& context, 
                                                   uint32_t length);
    std::vector<float> get_target_probabilities(const std::vector<llama_token>& context,
                                               const std::vector<llama_token>& candidates);
    std::vector<llama_token> verify_and_accept_tokens(const std::vector<llama_token>& draft_tokens,
                                                     const std::vector<float>& target_probs,
                                                     const std::vector<float>& draft_probs);
    
    // Probability analysis
    float calculate_acceptance_probability(float target_prob, float draft_prob);
    bool should_accept_token(float target_prob, float draft_prob);
    
    // Optimization helpers
    void update_context_with_accepted_tokens(const std::vector<llama_token>& accepted_tokens);
    void rollback_context_to_position(size_t position);

public:
    SpeculativeDecoder();
    ~SpeculativeDecoder();
    
    bool initialize(Ref<LlamaModel> p_target_model, llama_context* p_target_ctx,
                   Ref<LlamaModel> p_draft_model = Ref<LlamaModel>(), 
                   llama_context* p_draft_ctx = nullptr);
    void shutdown();
    
    // Main generation interface
    std::vector<llama_token> generate_tokens_speculative(const std::vector<llama_token>& prompt,
                                                        uint32_t max_tokens);
    
    // Single step generation for streaming
    llama_token generate_next_token_speculative(const std::vector<llama_token>& context);
    
    // Configuration
    void set_lookahead_tokens(uint32_t tokens);
    uint32_t get_lookahead_tokens() const;
    void set_acceptance_threshold(float threshold);
    float get_acceptance_threshold() const;
    void set_max_draft_length(uint32_t length);
    uint32_t get_max_draft_length() const;
    void set_use_separate_draft_model(bool use_separate);
    bool get_use_separate_draft_model() const;
    
    // Performance monitoring
    struct SpeculativeStats {
        uint64_t total_tokens_generated;
        uint64_t accepted_draft_tokens;
        uint64_t rejected_draft_tokens;
        uint64_t speculation_cycles;
        float acceptance_rate;
        float speedup_ratio;
        float average_accepted_length;
    };
    
    SpeculativeStats get_stats() const;
    void reset_stats();
    
    // Utility methods
    bool is_initialized() const;
    void update_sampling_parameters(float temperature, float top_p);
};

} // namespace godot

#endif