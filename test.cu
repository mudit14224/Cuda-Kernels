#include <iostream>

__global__ void addKernel(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    const int arraySize = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5};
    const int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};

    // Device arrays
    int *dev_a = nullptr;
    int *dev_b = nullptr;
    int *dev_c = nullptr;

    // Allocate device memory
    cudaMalloc((void **)&dev_a, arraySize * sizeof(int));
    cudaMalloc((void **)&dev_b, arraySize * sizeof(int));
    cudaMalloc((void **)&dev_c, arraySize * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    addKernel<<<1, arraySize>>>(dev_c, dev_a, dev_b);

    // Copy result back to host
    cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // Output result
    for (int i = 0; i < arraySize; ++i) {
        std::cout << c[i] << ' ';
    }
    std::cout << std::endl;

    return 0;
}
