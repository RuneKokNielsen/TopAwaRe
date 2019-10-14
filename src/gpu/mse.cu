
#include "gpu.h"
#include "mse.h"

__global__
void kernel_mse_alpha_sum(int n, float eta, float *g_a, float *g_b, float *g_c, float *g_out){

  extern __shared__ float sdata[];


  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  sdata[tid] = 0;
  if(i >= n) return;

  float d = g_a[i] - g_b[i];
  d = d * d * g_c[i] + (1 - g_c[i]) * eta;
  sdata[tid] = d;

  __syncthreads();
  
  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }


  // write result for this block to global mem
  if (tid == 0) g_out[blockIdx.x] = sdata[0];
  
}
float cuda_mse_alpha_sum(int n, float eta, float *g_a, float *g_b, float *g_c){ 
  
  float *g_o;
  cudaMalloc(&g_o, nblocks(n) * sizeof(float));
  cuda_error_handle();

  kernel_mse_alpha_sum<<<nblocks(n),blocksize(), blocksize() * sizeof(float)>>>(n, eta, g_a, g_b, g_c, g_o);
  cuda_error_handle();
  float res2[nblocks(n)];  
  cudaMemcpy(&res2, g_o, nblocks(n) * sizeof(float), cudaMemcpyDeviceToHost);
  cuda_error_handle();

  float res = 0;
  for(int i=0; i<nblocks(n); i++){
    res += res2[i];
  }

  cudaFree(g_o);
  cuda_error_handle();
  return res;
  
}
