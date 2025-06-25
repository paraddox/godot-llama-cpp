#include "speculative_decoder.h"
#include "../llama.cpp/common/common.h"
#include <algorithm>
#include <random>
#include <cmath>

using namespace godot;

SpeculativeDecoder::SpeculativeDecoder() {
    decoder_mutex.instantiate();
    target_ctx = nullptr;
    draft_ctx = nullptr;
    target_sampler = nullptr;
    draft_sampler = nullptr;
}

SpeculativeDecoder::~SpeculativeDecoder() {
    shutdown();
}

bool SpeculativeDecoder::initialize(Ref<LlamaModel> p_target_model, llama_context* p_target_ctx,
                                  Ref<LlamaModel> p_draft_model, llama_context* p_draft_ctx) {
    if (p_target_model.is_null() || p_target_model->model == nullptr || p_target_ctx == nullptr) {
        UtilityFunctions::printerr("SpeculativeDecoder: Invalid target model or context");
        return false;
    }
    
    target_model = p_target_model;
    target_ctx = p_target_ctx;
    
    // Set up draft model (use target model if not provided)
    if (p_draft_model.is_valid() && p_draft_model->model != nullptr && p_draft_ctx != nullptr) {
        draft_model = p_draft_model;
        draft_ctx = p_draft_ctx;
        use_separate_draft_model = true;
        UtilityFunctions::print("SpeculativeDecoder: Using separate draft model");
    } else {
        draft_model = target_model;
        draft_ctx = target_ctx;
        use_separate_draft_model = false;
        UtilityFunctions::print("SpeculativeDecoder: Using target model for drafting");
    }
    
    // Create samplers
    target_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(target_sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(target_sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(target_sampler, llama_sampler_init_dist(0));
    
    // For draft sampler, use slightly more aggressive sampling for diversity
    draft_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(draft_sampler, llama_sampler_init_temp(1.0f));  // Higher temp
    llama_sampler_chain_add(draft_sampler, llama_sampler_init_top_p(0.9f, 1)); // Broader top_p
    llama_sampler_chain_add(draft_sampler, llama_sampler_init_dist(0));
    
    UtilityFunctions::print("SpeculativeDecoder: Initialized successfully");
    return true;
}

void SpeculativeDecoder::shutdown() {
    decoder_mutex->lock();
    
    active_sequences.clear();
    context_tokens.clear();
    
    if (target_sampler) {
        llama_sampler_free(target_sampler);
        target_sampler = nullptr;
    }
    
    if (draft_sampler && draft_sampler != target_sampler) {
        llama_sampler_free(draft_sampler);
        draft_sampler = nullptr;
    }
    
    decoder_mutex->unlock();
}

std::vector<llama_token> SpeculativeDecoder::generate_tokens_speculative(
    const std::vector<llama_token>& prompt, uint32_t max_tokens) {
    
    if (!is_initialized()) {
        UtilityFunctions::printerr("SpeculativeDecoder: Not initialized");
        return {};
    }
    
    decoder_mutex->lock();
    context_tokens = prompt;
    std::vector<llama_token> generated_tokens;
    decoder_mutex->unlock();
    
    for (uint32_t i = 0; i < max_tokens; i++) {
        llama_token next_token = generate_next_token_speculative(context_tokens);
        if (next_token == 0) break; // End of generation
        
        generated_tokens.push_back(next_token);
        
        decoder_mutex->lock();
        context_tokens.push_back(next_token);
        decoder_mutex->unlock();
        
        // Check for end-of-generation token
        const struct llama_vocab* vocab = llama_model_get_vocab(target_model->model);
        if (llama_vocab_is_eog(vocab, next_token)) {
            break;
        }
    }
    
    return generated_tokens;
}

llama_token SpeculativeDecoder::generate_next_token_speculative(const std::vector<llama_token>& context) {
    if (!is_initialized()) {
        return 0;
    }
    
    speculation_cycles.fetch_add(1);
    
    // Step 1: Generate draft sequence
    std::vector<llama_token> draft_tokens = generate_draft_sequence(context, lookahead_tokens);
    
    if (draft_tokens.empty()) {
        // Fall back to regular sampling
        llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(context.data()), context.size());
        if (llama_decode(target_ctx, batch) != 0) {
            return 0;
        }
        
        llama_token token = llama_sampler_sample(target_sampler, target_ctx, -1);
        llama_sampler_accept(target_sampler, token);
        total_tokens_generated.fetch_add(1);
        return token;
    }
    
    // Step 2: Get target model probabilities for draft tokens
    std::vector<llama_token> context_with_drafts = context;
    std::vector<float> draft_probabilities;
    std::vector<float> target_probabilities;
    
    // Generate target probabilities for each draft token position
    for (size_t i = 0; i < draft_tokens.size(); i++) {
        // Decode context up to this position
        llama_batch batch = llama_batch_get_one(context_with_drafts.data(), context_with_drafts.size());
        if (llama_decode(target_ctx, batch) != 0) {
            break;
        }
        
        // Get probability for the draft token at this position
        // For simplicity, we'll use sampler probability estimation
        llama_token draft_token = draft_tokens[i];
        
        // Sample from target to get its preferred token and probability
        llama_token target_token = llama_sampler_sample(target_sampler, target_ctx, -1);
        
        // Calculate acceptance based on token match and probability ratio
        bool accept = (target_token == draft_token) || 
                     (static_cast<float>(rand()) / RAND_MAX < acceptance_threshold);
        
        if (accept) {
            context_with_drafts.push_back(draft_token);
            llama_sampler_accept(target_sampler, draft_token);
            accepted_draft_tokens.fetch_add(1);
        } else {
            // Reject this and all subsequent draft tokens
            rejected_draft_tokens.fetch_add(draft_tokens.size() - i);
            llama_sampler_accept(target_sampler, target_token);
            total_tokens_generated.fetch_add(1);
            return target_token;
        }
    }
    
    // If we get here, all draft tokens were accepted
    total_tokens_generated.fetch_add(draft_tokens.size());
    
    // Return the first accepted token (caller will call again for subsequent tokens)
    return draft_tokens.empty() ? 0 : draft_tokens[0];
}

std::vector<llama_token> SpeculativeDecoder::generate_draft_sequence(
    const std::vector<llama_token>& context, uint32_t length) {
    
    std::vector<llama_token> draft_tokens;
    std::vector<llama_token> working_context = context;
    
    for (uint32_t i = 0; i < length && i < max_draft_length; i++) {
        // Decode current context
        llama_batch batch = llama_batch_get_one(working_context.data(), working_context.size());
        if (llama_decode(draft_ctx, batch) != 0) {
            break;
        }
        
        // Sample next token with draft sampler (more aggressive)
        llama_token draft_token = llama_sampler_sample(draft_sampler, draft_ctx, -1);
        llama_sampler_accept(draft_sampler, draft_token);
        
        draft_tokens.push_back(draft_token);
        working_context.push_back(draft_token);
        
        // Check for end token
        const struct llama_vocab* vocab = llama_model_get_vocab(draft_model->model);
        if (llama_vocab_is_eog(vocab, draft_token)) {
            break;
        }
    }
    
    return draft_tokens;
}

std::vector<float> SpeculativeDecoder::get_target_probabilities(
    const std::vector<llama_token>& context, const std::vector<llama_token>& candidates) {
    
    std::vector<float> probabilities;
    probabilities.reserve(candidates.size());
    
    // This is a simplified implementation
    // In practice, you'd want to get the actual logits from the model
    for (size_t i = 0; i < candidates.size(); i++) {
        // Placeholder probability calculation
        probabilities.push_back(0.5f + static_cast<float>(rand()) / RAND_MAX * 0.5f);
    }
    
    return probabilities;
}

std::vector<llama_token> SpeculativeDecoder::verify_and_accept_tokens(
    const std::vector<llama_token>& draft_tokens,
    const std::vector<float>& target_probs,
    const std::vector<float>& draft_probs) {
    
    std::vector<llama_token> accepted_tokens;
    
    for (size_t i = 0; i < draft_tokens.size() && i < target_probs.size() && i < draft_probs.size(); i++) {
        if (should_accept_token(target_probs[i], draft_probs[i])) {
            accepted_tokens.push_back(draft_tokens[i]);
            accepted_draft_tokens.fetch_add(1);
        } else {
            rejected_draft_tokens.fetch_add(draft_tokens.size() - i);
            break;
        }
    }
    
    return accepted_tokens;
}

float SpeculativeDecoder::calculate_acceptance_probability(float target_prob, float draft_prob) {
    if (draft_prob <= 0.0f) return 0.0f;
    return std::min(1.0f, target_prob / draft_prob);
}

bool SpeculativeDecoder::should_accept_token(float target_prob, float draft_prob) {
    float acceptance_prob = calculate_acceptance_probability(target_prob, draft_prob);
    return (static_cast<float>(rand()) / RAND_MAX) < acceptance_prob;
}

void SpeculativeDecoder::update_context_with_accepted_tokens(const std::vector<llama_token>& accepted_tokens) {
    context_tokens.insert(context_tokens.end(), accepted_tokens.begin(), accepted_tokens.end());
}

void SpeculativeDecoder::rollback_context_to_position(size_t position) {
    if (position < context_tokens.size()) {
        context_tokens.resize(position);
    }
}

// Configuration methods
void SpeculativeDecoder::set_lookahead_tokens(uint32_t tokens) {
    lookahead_tokens = std::min(tokens, max_draft_length);
}

uint32_t SpeculativeDecoder::get_lookahead_tokens() const {
    return lookahead_tokens;
}

void SpeculativeDecoder::set_acceptance_threshold(float threshold) {
    acceptance_threshold = std::clamp(threshold, 0.0f, 1.0f);
}

float SpeculativeDecoder::get_acceptance_threshold() const {
    return acceptance_threshold;
}

void SpeculativeDecoder::set_max_draft_length(uint32_t length) {
    max_draft_length = std::max(1u, std::min(length, 16u)); // Reasonable bounds
}

uint32_t SpeculativeDecoder::get_max_draft_length() const {
    return max_draft_length;
}

void SpeculativeDecoder::set_use_separate_draft_model(bool use_separate) {
    use_separate_draft_model = use_separate;
}

bool SpeculativeDecoder::get_use_separate_draft_model() const {
    return use_separate_draft_model;
}

void SpeculativeDecoder::update_sampling_parameters(float temperature, float top_p) {
    // Update target sampler
    if (target_sampler) {
        llama_sampler_free(target_sampler);
        target_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(target_sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(target_sampler, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(target_sampler, llama_sampler_init_dist(0));
    }
    
    // Update draft sampler with slightly more aggressive parameters
    if (draft_sampler) {
        llama_sampler_free(draft_sampler);
        draft_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(draft_sampler, llama_sampler_init_temp(temperature * 1.2f));
        llama_sampler_chain_add(draft_sampler, llama_sampler_init_top_p(top_p * 0.9f, 1));
        llama_sampler_chain_add(draft_sampler, llama_sampler_init_dist(0));
    }
}

bool SpeculativeDecoder::is_initialized() const {
    return target_model.is_valid() && target_ctx != nullptr && 
           target_sampler != nullptr && draft_sampler != nullptr;
}

SpeculativeDecoder::SpeculativeStats SpeculativeDecoder::get_stats() const {
    SpeculativeStats stats;
    stats.total_tokens_generated = total_tokens_generated.load();
    stats.accepted_draft_tokens = accepted_draft_tokens.load();
    stats.rejected_draft_tokens = rejected_draft_tokens.load();
    stats.speculation_cycles = speculation_cycles.load();
    
    uint64_t total_draft_tokens = stats.accepted_draft_tokens + stats.rejected_draft_tokens;
    stats.acceptance_rate = total_draft_tokens > 0 ? 
        static_cast<float>(stats.accepted_draft_tokens) / total_draft_tokens * 100.0f : 0.0f;
    
    stats.speedup_ratio = stats.speculation_cycles > 0 ?
        static_cast<float>(stats.total_tokens_generated) / stats.speculation_cycles : 1.0f;
    
    stats.average_accepted_length = stats.speculation_cycles > 0 ?
        static_cast<float>(stats.accepted_draft_tokens) / stats.speculation_cycles : 0.0f;
    
    return stats;
}

void SpeculativeDecoder::reset_stats() {
    total_tokens_generated.store(0);
    accepted_draft_tokens.store(0);
    rejected_draft_tokens.store(0);
    speculation_cycles.store(0);
}