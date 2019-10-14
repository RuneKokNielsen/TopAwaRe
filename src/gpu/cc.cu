
#include "gpu.h"
#include "cc.h"

__global__
void kernel_cuda_cc_preproc(int n, float* M, float* Sinv, float* A, float* B, float* C, float* Mbar, float* Sinvbar, int dimx, int dimy, int dimz, int w){

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;

  float sum_M = 0;
  float sum_M_2 = 0;
  float sum_Sinv = 0;
  float sum_Sinv_2 = 0;
  float sum_M_Sinv = 0;
  int elements = 0;

  int nperrow = dimy * dimz;

  int x0 = i / nperrow;
  int y0 = (i % nperrow) / dimz;
  int z0 = i % dimz;

  for(int x=max(0, x0 - w); x<=min(dimx - 1, x0 + w); x++){
    for(int y=max(0, y0 - w); y<=min(dimy - 1, y0 + w); y++){
      for(int z=max(0, z0 - w); z<=min(dimz - 1, z0 + w); z++){
        int j = x * nperrow + y * dimz + z;
        sum_M += M[j];
        sum_M_2 += M[j] * M[j];
        sum_Sinv += Sinv[j];
        sum_Sinv_2 += Sinv[j] * Sinv[j];
        sum_M_Sinv += M[j] * Sinv[j];
        elements ++;
      }
    }
  }

  float mean_M = sum_M / elements;
  float mean_Sinv = sum_Sinv / elements;

  Mbar[i] = M[i] - mean_M;
  Sinvbar[i] = Sinv[i] - mean_Sinv;

  A[i] = sum_M_Sinv - mean_M * sum_Sinv - mean_Sinv * sum_M + elements * mean_M * mean_Sinv;
  B[i] = sum_M_2 - mean_M * sum_M - mean_M * sum_M + elements * mean_M * mean_M;
  C[i] = sum_Sinv_2 - mean_Sinv * sum_Sinv - mean_Sinv * sum_Sinv + elements * mean_Sinv * mean_Sinv;
  
}
void cuda_cc_preproc(int n, float* M, float* Sinv, float* A, float* B, float* C, float* Mbar, float* Sinvbar, int dimx, int dimy, int dimz, int w){

  kernel_cuda_cc_preproc<<<nblocks(n), blocksize()>>>(n, M, Sinv, A, B, C, Mbar, Sinvbar, dimx, dimy, dimz, w);
  cuda_error_handle();  
}



__global__
void kernel_cc_grad(int n, float* Mbar, float* Sinvbar, float* A, float* B, float* C, float* target){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;

  if(B[i] * C[i] == 0){
    target[i] = 0;
    return;
  }

  if(A[i] < 0.0001){
    target[i] = 0;
    return;
  }

  target[i] =  2 * A[i] / (B[i] * C[i]) * (Sinvbar[i] - A[i] / B[i] * Mbar[i]);
}
void cuda_cc_grad(int n, float* Mbar, float* Sinvbar, float* A, float* B, float* C, float* target){

  kernel_cc_grad<<<nblocks(n), blocksize()>>>(n, Mbar, Sinvbar, A, B, C, target);
  cuda_error_handle();
  
}

__global__
void kernel_cc_measure(int n, float* A, float* B, float* C, float* target){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;

  if(B[i] * C[i] == 0){
    target[i] = 1;
  }else{
    target[i] = A[i]*A[i] / (B[i] * C[i]);
  }
  
}
void cuda_cc_measure(int n, float* A, float* B, float* C, float* target){

  kernel_cc_measure<<<nblocks(n), blocksize()>>>(n, A, B, C, target);
  cuda_error_handle();
  
}


