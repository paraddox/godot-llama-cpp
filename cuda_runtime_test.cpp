#include <cuda_runtime.h>
#include <iostream>
int main() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    std::cout << "Device count: " << count << std::endl;
    if (count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device 0: " << prop.name << std::endl;
    }
    return 0;
}
