#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include <vector>

using namespace std;

__global__ void MatMulKernel(float *A, float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < N) && (col < N))
    {
        float val = 0;
        for(int k = 0; k < N; k++)
        {
            val += A[row*N+k] * B[k*N + col];
        }
        C[row*N+col] = val;
    }
}

void printMatrix(const vector<float>& matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i * N + j] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

int main()
{
    int N = 16;
    int SIZE = N*N;
    srand(time(NULL));
    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);
    vector<float> h_C(SIZE);

    for(int i = 0; i<N; i++)
    {
        for(int j = 0; j<N; j++)
        {
            h_A[i*N + j] = rand() % 100;
            h_B[i*N + j] = rand() % 100;
        }
    }

    cout << "Matrix A:" << endl;
    printMatrix(h_A, N);

    cout << "Matrix B:" << endl;
    printMatrix(h_B, N);

    // Allocate memory on the device and copy from host to device
    float *d_A, *d_B, *d_C;

    if(cudaMalloc(&d_A, sizeof(float)*SIZE) != cudaSuccess)
    {
        cout<<"Error in allocation of memory!";
        return 0;
    }
    if(cudaMalloc(&d_B, sizeof(float)*SIZE) != cudaSuccess)
    {
        cout<<"Error in allocation of memory!";
        cudaFree(d_A);
        return 0;
    }
    if(cudaMalloc(&d_C, sizeof(float)*SIZE) != cudaSuccess)
    {
        cout<<"Error in allocation of memory!";
        cudaFree(d_A);
        cudaFree(d_B);
        return 0;
    }

    if(cudaMemcpy(d_A, h_A.data(), sizeof(float)*SIZE, cudaMemcpyHostToDevice)!=cudaSuccess)
    {
        cout<<"Error in copying from host to device!";
        return 0;
    }
    if(cudaMemcpy(d_B, h_B.data(), sizeof(float)*SIZE, cudaMemcpyHostToDevice)!=cudaSuccess)
    {
        cout<<"Error in copying from host to device!";
        cudaFree(d_A);
        return 0;
    }

    // run the cuda kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    MatMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();


    // copy from device to host and free memory
    if(cudaMemcpy(h_C.data(), d_C, sizeof(float)*SIZE, cudaMemcpyDeviceToHost)!=cudaSuccess)
    {
        cout<<"Error in copying from device to host!";
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    cout << "Result Matrix C:" << endl;
    printMatrix(h_C, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}