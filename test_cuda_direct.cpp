#include <iostream>
#include <ggml-cuda.h>

int main() {
    std::cout << "Direct CUDA test..." << std::endl;
    
    int count = ggml_backend_cuda_get_device_count();
    std::cout << "ggml_backend_cuda_get_device_count() = " << count << std::endl;
    
    if (count > 0) {
        std::cout << "CUDA devices available!" << std::endl;
        
        // Get device info
        char desc[256];
        ggml_backend_cuda_get_device_description(0, desc, sizeof(desc));
        std::cout << "Device 0: " << desc << std::endl;
    } else {
        std::cout << "No CUDA devices found" << std::endl;
    }
    
    return 0;
}