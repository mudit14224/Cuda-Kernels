#include <iostream>
#include <cuda_runtime.h>

// compute S function (S => QK^T)
// Q,K => (Nxd)
// S => (NxN)
// N: Sequence length
// d: head dimension
__global__ void compute_S(float *Q, float *K, float *S, int N, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        // compute S = Q * K^T
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            sum += Q[row * d + i] * K[col * d + i]
        }
        S[row * N + col] = sum;
    }
}

// Compute P => Softmax(S) (This is performed row wise)
// S: NxN
// P: NxN
__global__ void softmax(float *S, float *P, int N) {
    int row = blockIdx.x + blockDim.x + threadIdx.x;
    if (row < N) {
        float max_val = -INFINITY; 
        // find the max val in order to avoid overflow during softmax
        for (int col=0; col<N; col++) {
            max_val = fmaxf(max_val, S[row * N + col]);
        }

        // Compute softmax
        float sum_exp = 0.0f; 
        for (int col = 0; col < N; col++) {
            sum_exp += expf(S[row * N + col] - max_val);
        }

        for (int col=0; col<N; col++) {
            P[row * N + col] = expf(S[row * N + col] - max_val) / sum_exp; 
        }
    }
}

// Compute O => P*V
// P: NxN
// V: Nxd
// O: Nxd
__global__ void compute_O(float *P, float *V, float *O, int N, int d) {
    row = blockIdx.x + blockDim.x + threadIdx.x; 
    col = blockIdx.y + blockDim.y + threadIdx.y;

    if (row < N && col < d) {
        float sum = 0.0f;
        for (int i=0; i<N; i++){
            sum += P[row * N + i] * V[i * d + col]
        }
        O[row * d + col] = sum;
    }
}

// Attention Kerner
void attention_kernel(float *Q, float *K, float *V, float *O, int N, int d) {
    float *d_Q, *d_K, *d_S, *d_P, *d_V, *d_O;

    // Allocate memory on the device (GPU)
    cudaMalloc((void**)&d_Q, N * d * sizeof(float)); 
    cudaMalloc((void**)&d_K, N * d * sizeof(float));
    cudaMalloc((void**)&d_S, N * N * sizeof(float));
    cudaMalloc((void**)&d_P, N * N * sizeof(float));
    cudaMalloc((void**)&d_V, N * d * sizeof(float));
    cudaMalloc((void**)&d_O, N * d * sizeof(float));

    // Copy the data to device (GPU)
    // cudaMemcpy(to, from, size, typeOfTransfer)
    cudaMemcpy(d_Q, Q, N * d * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_K, K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * d * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16) // 16x16 threads
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Step 1: Compute S = Q * K^T
    compute_S<<<gridSize, blockSize>>>(d_Q, d_K, d_S, N, d);
    cudaDeviceSynchronize(); 

    // Step 2: Compute softmax(P) from S
    // For softmax we process each row independetly 
    // which is why the grid size is 1D in this case
    softmax<<<(N + blockSize.x - 1) / blockSize.x, blockSize.x>>>(d_S, d_P, N);
    cudaDeviceSynchronize();

    // Step 3: Compute O = P * V
    compute_O<<<gridSize, blockSize>>>(d_P, d_V, d_O, N, d);
    cudaDeviceSynchronize();

    // Step 4: Copy result O back to host
    cudaMemcpy(O, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_S);
    cudaFree(d_P);
    cudaFree(d_V);
    cudaFree(d_O);
}

int main() {
    int N = 1024; // Example dimension
    int d = 512;  // Example dimension

    // Allocate matrices Q, K, V, O on host
    float *Q = new float[N * d];
    float *K = new float[N * d];
    float *V = new float[N * d];
    float *O = new float[N * d];

     // Initialize matrices Q, K, V (we will use random values)
    for (int i = 0; i < N * d; i++) {
        Q[i] = static_cast<float>(rand()) / RAND_MAX;
        K[i] = static_cast<float>(rand()) / RAND_MAX;
        V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Call the attention kernel function
    attention_kernel(Q, K, V, O, N, d);

    // Output result O (we just print a small portion)
    std::cout << "Resulting O[0, 0]: " << O[0] << std::endl;

    // Clean up
    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] O;

    return 0;
}



