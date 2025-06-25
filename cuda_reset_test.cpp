#include <cuda_runtime.h>
#include <iostream>
int main() {
    // Force reset any existing CUDA context
    cudaDeviceReset();
    
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    std::cout << "CUDA Error after reset: " << cudaGetErrorString(err) << std::endl;
    std::cout << "Device count: " << count << std::endl;
    
    if (err == cudaSuccess && count > 0) {
        std::cout << "CUDA is working\!" << std::endl;
        return 0;
    } else {
        std::cout << "CUDA failed" << std::endl;
        return 1;
    }
}
