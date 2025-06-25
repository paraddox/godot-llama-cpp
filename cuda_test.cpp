#include <iostream>
#include <ggml-backend.h>
#include <ggml-cuda.h>

int main() {
    std::cout << "Testing CUDA backend detection..." << std::endl;
    
    // Load all backends
    std::cout << "Calling ggml_backend_load_all()..." << std::endl;
    ggml_backend_load_all();
    
    // Check CUDA device count
    std::cout << "Checking CUDA device count..." << std::endl;
    int cuda_device_count = ggml_backend_cuda_get_device_count();
    std::cout << "CUDA device count: " << cuda_device_count << std::endl;
    
    // Check backend registrations
    size_t backend_count = ggml_backend_reg_count();
    std::cout << "Total backend registrations: " << backend_count << std::endl;
    
    for (size_t i = 0; i < backend_count; i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        const char* name = ggml_backend_reg_name(reg);
        std::cout << "  Backend " << i << ": " << name << std::endl;
    }
    
    return 0;
}