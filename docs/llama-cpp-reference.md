# llama.cpp C++ API Reference

This file contains API reference documentation for llama.cpp, focused on the C++ bindings used in this project.

## Project Overview

llama.cpp is a C/C++ library for running LLaMA and other language models locally with minimal setup and state-of-the-art performance.

Key features:
- Plain C/C++ implementation without dependencies
- Apple Silicon optimized (ARM NEON, Accelerate, Metal)
- AVX, AVX2, AVX512, AMX support for x86
- 1.5-bit to 8-bit integer quantization
- CUDA, HIP, Vulkan, SYCL backend support
- CPU+GPU hybrid inference

## Recent API Changes

⚠️ **CRITICAL**: The project maintains a [changelog for libllama API](https://github.com/ggml-org/llama.cpp/issues/9289) due to frequent breaking changes.

### Key Breaking Changes (affecting our integration):

1. **Token Checking Functions**:
   ```cpp
   // OLD API (removed)
   bool llama_token_is_eog(const struct llama_model * model, llama_token token);
   
   // NEW API 
   bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);
   // Usage: llama_vocab_is_eog(llama_model_get_vocab(model), token)
   ```

2. **Context Creation** (deprecated but still functional):
   ```cpp
   // DEPRECATED
   struct llama_context * llama_new_context_with_model(
       struct llama_model * model,
       struct llama_context_params params);
   
   // NEW (recommended)
   struct llama_context * llama_init_from_model(
       struct llama_model * model, 
       struct llama_context_params params);
   ```

3. **KV Cache Management**:
   ```cpp
   // REMOVED - no direct replacement
   void llama_kv_cache_seq_rm(
       struct llama_context * ctx,
       llama_seq_id seq_id,
       llama_pos p0,
       llama_pos p1);
   ```

4. **Tokenization API**:
   ```cpp
   // OLD API
   int llama_tokenize(
       struct llama_context * ctx,
       const char * text,
       llama_token * tokens,
       int n_max_tokens,
       bool add_bos,
       bool special);
   
   // NEW API
   int32_t llama_tokenize(
       const struct llama_vocab * vocab,
       const char * text,
       int32_t text_len,
       llama_token * tokens,
       int32_t n_max_tokens,
       bool add_special,
       bool parse_special);
   ```

5. **Token to Piece**:
   ```cpp
   // OLD API 
   std::string llama_token_to_piece(
       const struct llama_model * model,
       llama_token token);
   
   // NEW API
   int32_t llama_token_to_piece(
       const struct llama_vocab * vocab,
       llama_token token,
       char * buf,
       int32_t length,
       bool lstrip,
       bool special);
   ```

6. **Sampling Context** (completely restructured):
   ```cpp
   // OLD API (removed)
   struct llama_sampling_context;
   struct llama_sampling_params;
   
   // NEW API 
   struct llama_sampler;
   // Sampling is now handled through sampler chains
   ```

## Core Data Structures

### Basic Types
```cpp
typedef int32_t llama_pos;     // Position in sequence
typedef int32_t llama_token;   // Token ID
typedef int32_t llama_seq_id;  // Sequence ID
```

### Model Parameters
```cpp
struct llama_model_params {
    ggml_backend_dev_t * devices;              // GPU devices for offloading
    int32_t n_gpu_layers;                      // layers to store in VRAM
    enum llama_split_mode split_mode;          // how to split across GPUs
    int32_t main_gpu;                          // primary GPU when not splitting
    const float * tensor_split;               // proportion per GPU
    llama_progress_callback progress_callback; // loading progress callback
    void * progress_callback_user_data;
    bool vocab_only;    // only load vocabulary
    bool use_mmap;      // use memory mapping
    bool use_mlock;     // force model in RAM
    bool check_tensors; // validate tensor data
};
```

### Context Parameters
```cpp
struct llama_context_params {
    uint32_t n_ctx;             // text context size
    uint32_t n_batch;           // logical batch size
    uint32_t n_ubatch;          // physical batch size  
    uint32_t n_seq_max;         // max sequences
    int32_t  n_threads;         // threads for generation
    int32_t  n_threads_batch;   // threads for batch processing
    
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type      pooling_type;
    enum llama_attention_type    attention_type;
    
    float rope_freq_base;   // RoPE base frequency
    float rope_freq_scale;  // RoPE frequency scaling
    // ... additional RoPE and YaRN parameters
};
```

### Batch Structure
```cpp
typedef struct llama_batch {
    int32_t n_tokens;

    llama_token  *  token;    // token IDs (when embd is NULL)
    float        *  embd;     // embeddings (when token is NULL)
    llama_pos    *  pos;      // token positions
    int32_t      *  n_seq_id; // number of sequences per token
    llama_seq_id ** seq_id;   // sequence IDs
    int8_t       *  logits;   // output control flags
} llama_batch;
```

## Key Functions

### Model and Context Management
```cpp
// Model loading
struct llama_model * llama_load_model_from_file(
    const char * path_model,
    struct llama_model_params params);

void llama_free_model(struct llama_model * model);

// Context creation  
struct llama_context * llama_new_context_with_model(
    struct llama_model * model,
    struct llama_context_params params);

void llama_free(struct llama_context * ctx);
```

### Inference
```cpp
// Decode tokens
int llama_decode(
    struct llama_context * ctx,
    struct llama_batch batch);

// Get logits
float * llama_get_logits_ith(
    struct llama_context * ctx, 
    int32_t i);
```

### Tokenization (Modern API)
```cpp
// Get vocabulary from model
const struct llama_vocab * llama_model_get_vocab(
    const struct llama_model * model);

// Tokenize text
int32_t llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_max_tokens,
    bool add_special,
    bool parse_special);

// Convert token to text
int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token,
    char * buf,
    int32_t length,
    bool lstrip,
    bool special);
```

### Sampling (Modern API)
```cpp
// Create sampler chain
struct llama_sampler * llama_sampler_chain_init(
    struct llama_sampler_chain_params params);

// Add samplers to chain
void llama_sampler_chain_add(
    struct llama_sampler * chain,
    struct llama_sampler * smpl);

// Sample token
llama_token llama_sampler_sample(
    struct llama_sampler * smpl,
    struct llama_context * ctx,
    int32_t idx);

// Accept token
void llama_sampler_accept(
    struct llama_sampler * smpl,
    llama_token token);
```

### Batch Operations
```cpp
// Initialize batch
struct llama_batch llama_batch_init(
    int32_t n_tokens,
    int32_t embd,
    int32_t n_seq_max);

// Add token to batch
void llama_batch_add(
    struct llama_batch * batch,
    llama_token id,
    llama_pos pos,
    const llama_seq_id * seq_ids,
    bool logits);

// Clear batch
void llama_batch_clear(struct llama_batch * batch);

// Free batch
void llama_batch_free(struct llama_batch batch);
```

## Integration Notes for Godot Extension

### Memory Management
- Always call `llama_free()` and `llama_free_model()` in destructors
- Use `llama_backend_init()` once at startup
- Call `llama_backend_free()` at shutdown

### Threading Considerations
- llama.cpp is generally thread-safe for read operations
- Use separate contexts for concurrent inference
- Batch processing can be parallelized with `n_threads_batch`

### Performance Tips
- Use appropriate `n_ctx` size (context window)
- Optimize `n_batch` and `n_ubatch` for your hardware
- Consider GPU offloading with `n_gpu_layers`
- Enable mmap with `use_mmap` for faster loading

## Migration Guide

For updating existing code to newer llama.cpp versions:

1. Replace `llama_token_is_eog()` with `llama_vocab_is_eog()`
2. Update tokenization calls to use vocab-based API
3. Replace sampling context with sampler chains
4. Remove `llama_kv_cache_seq_rm()` calls (no direct replacement)
5. Update `llama_token_to_piece()` to new signature

## Links

- [Official API Changelog](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Examples Directory](https://github.com/ggerganov/llama.cpp/tree/master/examples)
- [Build Documentation](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)