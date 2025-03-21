#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

// Q, K, V -> Nxd
#define M 1024 // On chip size M (SRAM)
#define N 1024 // seq len
#define d 512 // head size

#define Bc (M / (4 * d)) // block size for K and V
#define Br (min(M / (4 * d), d)) // block size for Q


// Kernel for FlashAttention
__global__ void flashattention_forward_kernel(float *Q, float *K, float *V, float *O, float *l, float *m, int N, int d, int Br, int Bc) {
    // Thread block indices
    int row_block = blockIdx.x;
    int col_block = blockIdx.y;

    // Thread indices within the block
    int thread_row = threadIdx.x;
    int thread_col = threadIdx.y;

    // Allocate shared memory for sub-blocks of Q, K, and V
    __shared__ float shared_Q[Br][d];
    __shared__ float shared_V[Bc][d];
    __shared__ float shared_K[Bc][d];

    // Algo (step 8): Load Q, K, V from global memory to shared memory 
    if (thread_row < Br && thread_col < d) {
        shared_Q[thread_row][thread_col] = Q[(row_block * Br + thread_row) * d + thread_col]; 
    }
    if (thread_row < Bc && thread_col < d) {
        shared_V[thread_row][thread_col] = V[(col_block * Bc + thread_row) * d + thread_col];
        shared_K[thread_row][thread_col] = K[(col_block * Bc + thread_row) * d + thread_col];
    }
    __syncthreads(); // Ensure all data is loaded into shared memory

    // Algo (step 9): Compute S = Q * K^T (on chip)
    float S[Br][Bc] = {0} // Attention scores
    for (int i = 0; i < d; i++) {
        for (int row = thread_row; row < Br; row += blockDim.x) {
            for (int col = thread_col; col < Bc; col += blockDim.y) {
                S[row][col] += shared_Q[row][i] * shared_K[col][i];
            }
        }
    }
    __syncthreads(); 

    // Algo (step 10): Compute row-wise max (m~), exp(S - m~) for P~ (softmax), and row sum for l~
    float row_max[Br] = {-INFINITY};
    float P[Br][Bc];
    for (int row = thread_row; row < Br; row += blockDim.x) {
        for (int col = thread_col; col < Bc; col += blockDim.y) {
            row_max[row] = fmaxf(row_max[row], S[row][col]);
        }
    }
    __syncthreads(); 

    for (int row = thread_row; row < Br; row += blockDim.x) {
        for (int col = thread_col; col < Bc; col += blockDim.y) {
            P[row][col] = expf(S[row][col] - row_max[row]);
        }
    }
    __syncthreads(); 

    // Algo (step 11): Compute new m and l
    float new_m[Br]; 
    float new_l[Br];
    for (int row = thread_row; row < Br; row += blockDim.x) {
        // max of current m and m~
        new_m[row] = fmax(m[row], row_max[row])
        // compute new l using weighted sum
        new_l[row] = expf(m[row] - new_m[row]) * l[row] + expf(row_max[row] - new_m[row] * row_sum[P[row]]);
    } 
    __syncthreads(); 

    // Algo (step 12): Write O back to the HBM
    for (int row = thread_row; row < Br; row += blockDim.x) {
        for (int col = thread_col; col < Bc; col += blockDim.y) {
            O[(row_block * Br + row) * d + col] = new_l[row] * shared_V[row][col]; 
        }
    }
    __syncthreads(); 

    // Algo (step 13): Write new m and new l to HBM
    for (int row = thread_row; row < Br; row += blockDim.x) {
        m[row] = new_m[row];
        l[row] = new_l[row];
    }
}

int main() {
    // Matrix dimensions
    int N = 1024;  // Number of rows/columns
    int d = 512;   // Dimension of Q, K, V matrices

    // Allocate matrices on host
    float *Q = new float[N * d];
    float *K = new float[N * d];
    float *V = new float[N * d];
    float *O = new float[N * d];
    float *l = new float[N];
    float *m = new float[N];

    // Initialize matrices Q, K, V (random values for demonstration)
    for (int i = 0; i < N * d; i++) {
        Q[i] = static_cast<float>(rand()) / RAND_MAX;
        K[i] = static_cast<float>(rand()) / RAND_MAX;
        V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on the device (GPU)
    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    cudaMalloc((void**)&d_Q, N * d * sizeof(float));
    cudaMalloc((void**)&d_K, N * d * sizeof(float));
    cudaMalloc((void**)&d_V, N * d * sizeof(float));
    cudaMalloc((void**)&d_O, N * d * sizeof(float));
    cudaMalloc((void**)&d_l, N * sizeof(float));
    cudaMalloc((void**)&d_m, N * sizeof(float));

    // Copy data to the device (GPU)
    cudaMemcpy(d_Q, Q, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * d * sizeof(float), cudaMemcpyHostToDevice);

    // Set block size and grid size for kernel launch
    dim3 blockSize(16, 16);  // Block size: 16x16 threads
    dim3 gridSize((N + Br - 1) / Br, (N + Bc - 1) / Bc) // Grid size: based on matrix dimensions

    // Launch FlashAttention kernel
    flashattention_kernel<<<gridSize, blockSize>>>(d_Q, d_K, d_V, d_O, d_l, d_m, N, d, Br, Bc);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(O, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost);

    // Output some of the result (for testing)
    std::cout << "Resulting O[0, 0]: " << O[0] << std::endl;

    // Free device memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_l);
    cudaFree(d_m);

    // Free host memory
    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] O;
    delete[] l;
    delete[] m;

    return 0;
}