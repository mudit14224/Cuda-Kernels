#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>

using namespace std;

__global__ void addKernel(int *a, int *b, int count)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < count)
    {
        a[id] += b[id];
    } 
}

int main()
{
    srand(time(NULL));
    int count = 1000;
    int *h_a = new int[count];
    int *h_b = new int[count];

    for(int i=0; i < count; i++)
    {
        h_a[i] = rand() % 1000;
        h_b[i] = rand() % 1000;
    }

    cout << "Before addition the top 5 numbers are: "<<endl;
    for(int i=0; i<5; i++)
    {
        cout<<h_a[i]<<" "<<h_b[i]<<endl;
    }

    // Allocate memory and copy to device from host
    int *d_a, *d_b;
    if(cudaMalloc(&d_a, sizeof(int) * count) != cudaSuccess)
    {
        cout<<"Error in memory allocation!";
        return 0;
    }

    if(cudaMalloc(&d_b, sizeof(int) * count) != cudaSuccess)
    {
        cout<<"Error in memory allocation!";
        cudaFree(d_a);
        return 0;
    }

    if(cudaMemcpy(d_a, h_a, sizeof(int) * count, cudaMemcpyHostToDevice)!=cudaSuccess)
    {
        cout<<"Error in copying the memory to device!";
        cudaFree(d_a);
        cudaFree(d_b);
        return 0;
    }

    if(cudaMemcpy(d_b, h_b, sizeof(int) * count, cudaMemcpyHostToDevice)!=cudaSuccess)
    {
        cout<<"Error in copying the memory to device!";
        cudaFree(d_a);
        cudaFree(d_b);
        return 0;
    }
    
    // run the cuda kernel
    addKernel<<<count / 256 + 1, 256>>>(d_a, d_b, count);

    // copy from device to host and free memory
    if(cudaMemcpy(h_a, d_a, sizeof(int) * count, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cout<<"Error in copying memory to host!";
        cudaFree(d_a);
        cudaFree(d_b);
        delete[] h_a;
        delete[] h_b;
        return 0;
    }

    cout<<"First 5 numbers after addition are: "<<endl;
    for(int i=0; i<5; i++)
    {
        cout<<h_a[i]<<endl;
    }

    delete[] h_a;
    delete[] h_b;
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
} 
