#
#include "gpu.h"
#include <stdio.h>
#include <unistd.h>

void cuda_set_device(int devid){
   cudaSetDevice(devid);
   cuda_error_handle();

   cudaDeviceReset();
   cuda_error_handle();

   /**
    * To prepare the device, shortly allocate most
    * available memory. There is no mention of this
    * in the cuda docs, but it prevents initialization
    * issues when the device is coming up from sleep.
    **/
   size_t free;
   size_t total;
   cudaMemGetInfo(&free, &total);
   char *data;
   
   cudaMalloc(&data, free * 0.9);
   cuda_error_handle();

   cudaFree(data);
   cuda_error_handle();

   
}

/**
 * Helpers
 **/
__global__
void kernel_zerofill(int n, float *a){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) a[i] = 0;
}
float *cuda_f32_allocate(int n, bool zerofill){
  float *g_a;
  cudaMalloc(&g_a, n * sizeof(float));
  cudaError_t errSync  = cudaGetLastError();
  while(errSync == 2){
    printf("Not enough memory available! Stalling...\n");
    cudaMalloc(&g_a, n * sizeof(float));
    errSync  = cudaGetLastError();
    usleep(10000000);
  }
  if(zerofill) kernel_zerofill<<<n/100,256>>>(n, g_a);
  cuda_error_handle();
  return g_a;
}

void cuda_f32_free(float *g_a){
  cudaFree(g_a);
  cuda_error_handle();
}

void cuda_f32_send(int n, float *a, float *g_a){
  cudaMemcpy(g_a, a, n*sizeof(float), cudaMemcpyHostToDevice);
  cuda_error_handle();
}

void cuda_f32_retrieve(int n, float *a, float *g_a){
  cudaMemcpy(a, g_a, n*sizeof(float), cudaMemcpyDeviceToHost);
  cuda_error_handle();
}

void cuda_error_handle(){
  cudaError_t errSync  = cudaGetLastError();
  if (errSync != cudaSuccess){
    char buffer[50];
    sprintf(buffer, "Sync kernel error (%i): %s\n", errSync, cudaGetErrorString(errSync));
    throw CudaException(buffer);
  }
  cudaError_t errAsync = cudaDeviceSynchronize();  
  if (errAsync != cudaSuccess){
    char buffer[50];
    sprintf(buffer, "Async kernel error (%i): %s\n", errAsync, cudaGetErrorString(errAsync));
    throw CudaException(buffer);
  }
   
}




/**
 * Internal helpers
 **/
int blocksize(){ return CUDA_BLOCKSIZE; }
int nblocks(int n){ return (n + blocksize()-1) / blocksize(); }
