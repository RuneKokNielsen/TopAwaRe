#include "gpu.h"
#include "image.h"

/**
 * Data structure change
 **/

__global__
void kernel_complex_real(int n, float *a, float *b){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) b[i] = a[i*2];
}
void cuda_complex_real(int n, cufftComplex *a, float *b){
  kernel_complex_real<<<nblocks(n),blocksize()>>>(n, (float*) a, b);
  cuda_error_handle();
}


__global__
void kernel_real_complex(int n, float *a, float *b){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n){
    b[i*2] = a[i];
    b[i*2+1] = 0;
  }
}
void cuda_real_complex(int n, float *a, cufftComplex *b){
  kernel_real_complex<<<nblocks(n),blocksize()>>>(n, a, (float*) b);
  cuda_error_handle();
}





/**
 * ==================================================================
 * Common operations
 * ==================================================================
 **/

/**
 * ADDITION
 **/
__global__
void kernel_madd(int n, float *a, float *b, float *c){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];  
}
void cuda_madd(int n, float *g_a, float *g_b, float *g_c){
  kernel_madd<<<nblocks(n),blocksize()>>>(n, g_a, g_b, g_c);
  cuda_error_handle();
}


/**
 * SUBTRACTION
 **/
__global__
void kernel_msub(int n, float *a, float *b, float *c){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] - b[i];  
}
void cuda_msub(int n, float *g_a, float *g_b, float *g_c){
  kernel_msub<<<nblocks(n),blocksize()>>>(n, g_a, g_b, g_c);
  cuda_error_handle();
}

/**
 * ELEMENT-WISE MULTIPLICATION
 **/
__global__
void kernel_mult(int n, float *a, float *b, float *c){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] * b[i];  
}
void cuda_mult(int n, float *g_a, float *g_b, float *g_c){
  kernel_mult<<<nblocks(n),blocksize()>>>(n, g_a, g_b, g_c);
  cuda_error_handle();
}

/**
 * COMPLEX ELEMENT-WISE MULTIPLICATION
 **/
__global__
void kernel_mult_cpx(int n, float *a, float *b, float *c){
  int i = (blockIdx.x*blockDim.x + threadIdx.x);
  if(i < n){
    int i2 = i * 2;
    c[i2] = a[i2] * b[i2] - a[i2+1] * b[i2+1];
    c[i2+1] = a[i2] * b[i2+1] + a[i2+1] * b[i2];
  }
}
void cuda_mult_cpx(int n, cufftComplex *g_a, cufftComplex *g_b, cufftComplex *g_c){
  kernel_mult_cpx<<<nblocks(n),blocksize()>>>(n, (float*) g_a, (float*) g_b, (float*) g_c);
  cuda_error_handle();
}

/**
 * ELEMENT-WISE DIVISION
 **/
__global__
void kernel_div(int n, float *a, float *b, float *c){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;
  c[i] = a[i] / b[i];
}
void cuda_div(int n, float *g_a, float *g_b, float *g_c){
  kernel_div<<<nblocks(n),blocksize()>>>(n, g_a, g_b, g_c);
  cuda_error_handle();
}

/**
 * COMPLEX ELEMENT-WISE DIVISION
 **/
__global__
void kernel_div_cpx(int n, float *a, float *b, float *c){
  int i = (blockIdx.x*blockDim.x + threadIdx.x);
  if(i < n){
    int i2 = i * 2;
    float denom = b[i2]*b[i2] + b[i2+1]*b[i2+1];    
    c[i2] = (a[i2]*b[i2] + a[i2+1]*b[i2+1]) / denom;
    c[i2+1] = (a[i2+1]*b[i2] - a[i2]*b[i2+1]) / denom;
  }
}
void cuda_div_cpx(int n, cufftComplex *g_a, cufftComplex *g_b, cufftComplex *g_c){
  kernel_div_cpx<<<nblocks(n),blocksize()>>>(n, (float*) g_a, (float*) g_b, (float*) g_c);
  cuda_error_handle();
}

/**
 * SCALAR MULTIPLICATION
 **/
__global__
void kernel_scalar(int n, float *a, float s, float *b){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) b[i] = a[i] * s;
}
void cuda_scalar(int n, float *g_a, float s, float *g_b){
  kernel_scalar<<<nblocks(n),blocksize()>>>(n, g_a, s, g_b);
  cuda_error_handle();
}



/**
 * SCALAR ADDITION
 **/
__global__
void kernel_sadd(int n, float *a, float s, float *b){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) b[i] = a[i] + s;
}
void cuda_sadd(int n, float *g_a, float s, float *g_b){
  kernel_sadd<<<nblocks(n),blocksize()>>>(n, g_a, s, g_b);
  cuda_error_handle();
}

/**
 * ELEMENT MAXIMUM
 **/
__global__
void kernel_max(int n, float *a, float s, float *b){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) b[i] = max(a[i], s);
}
void cuda_max(int n, float *g_a, float *g_b, float min){
  kernel_max<<<nblocks(n),blocksize()>>>(n, g_a, min, g_b);
  cuda_error_handle();
}


/**
 * ==================================================================
 * Other operations
 * ==================================================================
 **/
/**
 * COMPUTE GRADIENT
 **/
__global__
void kernel_gradient(int n, float *a, float *b, int dlength, int vdim){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n){
    int idi = (i / dlength) % vdim;
    float t = 0 < idi && idi < vdim - 2 ? 2 : 1;
    float l = idi > 0 ? a[i - dlength] : a[i];
    float r = idi < vdim - 2 ? a[i + dlength] : a[i];
    b[i] = (r - l) / t;
  }
}
void cuda_gradient(int n, float *g_a, float *g_b, int dlength, int vdim){
  kernel_gradient<<<nblocks(n),blocksize()>>>(n, g_a, g_b, dlength, vdim);
  cuda_error_handle();
}


/**
 * COMPUTE 2x2 DETERMINANT
 **/
__global__
void kernel_2x2_determinant(int n, float *a0, float *a1, float *a2, float *a3, float *b){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n){
    b[i] = a0[i] * a3[i] - a1[i] * a2[i];
  }
}
void cuda_2x2_determinant(int n, float **g_as, float *g_b){
  kernel_2x2_determinant<<<nblocks(n),blocksize()>>>(n, g_as[0], g_as[1], g_as[2], g_as[3], g_b);
  cuda_error_handle();
}


/**
 * COMPUTE 3x3 DETERMINANT
 **/
__global__
void kernel_3x3_determinant(int n, float *a0, float *a1, float *a2, float *a3, float *a4,
			    float *a5, float *a6, float *a7, float *a8, float *b){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n){
    b[i] = a0[i] * a4[i] * a8[i]
      + a1[i] * a5[i] * a6[i]
      + a2[i] * a3[i] * a7[i]
      - a2[i] * a4[i] * a6[i]
      - a1[i] * a3[i] * a8[i]
      - a0[i] * a5[i] * a7[i];
  }
}

void cuda_3x3_determinant(int n, float **g_as, float *g_b){
  kernel_3x3_determinant<<<nblocks(n),blocksize()>>>(n, g_as[0], g_as[1], g_as[2], g_as[3], g_as[4], g_as[5], g_as[6], g_as[7], g_as[8], g_b);
  cuda_error_handle();
}

/**
 * COMPUTE MESHGRID
 **/
__global__
void kernel_2d_meshgrid(int n, float *ar, float *ac, float deltar, float deltac, int dimc){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n){
    ar[i] = (0.5 + i/dimc) * deltar;
    ac[i] = (0.5 + i % dimc) * deltac;
  }
}
__global__
void kernel_3d_meshgrid(int n, float *ar, float *ac, float *az, float deltar, float deltac, float deltaz, int dimz, int dimcz){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n){
    ar[i] = (0.5 + i / dimcz) * deltar;
    ac[i] = (0.5 + (i % dimcz) / dimz) * deltac;
    az[i] = (0.5 + i % dimz) * deltaz;
  }
}

void cuda_meshgrid(int n, float **g_as, uint d, uint *dims, float *deltas){
  switch(d){
  case 2:
    kernel_2d_meshgrid<<<nblocks(n),blocksize()>>>(n, g_as[0], g_as[1], deltas[0], deltas[1], dims[1]);
    break;
  case 3:
    kernel_3d_meshgrid<<<nblocks(n),blocksize()>>>(n, g_as[0], g_as[1], g_as[2], deltas[0], deltas[1], deltas[2], dims[2], dims[1] * dims[2]);
  };
  cuda_error_handle();
}




/**
 * COMPUTE MESHGRID
 **/
__global__
void kernel_3d_linear_interpolation(int n, float *in, float *out, float *dr, float* dc, float *dz, int nperrow, int dimr, int dimc, int dimz){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n){
    // Scale coordinates up to N-by-M
    double rlm = dr[i] * dimr;
    double clm = dc[i] * dimc;
    double zlm = dz[i] * dimz;


    int ra, ca, rd, cd, za, zd;
    float wr, wc, wz;
    // Get surrounding pixel coordinates
    if(rlm < 0.5){ // upper part of first row
      ra = 0; rd = ra; wr = 0;
    }else if(rlm >= dimr-0.5){ // lower part of last row
      ra = dimr-1; rd = ra; wr = 0;
    }else{ // interpolate between rows
      ra = floor(rlm - 0.5); rd = ra + 1; wr = (rlm - 0.5) - ra;
    }

    if(clm < 0.5){ // left part of first col
      ca = 0; cd = ca; wc = 0;
    }else if(clm >= dimc-0.5){ // right part of last col
      ca = dimc-1; cd = ca; wc = 0;
    }else{ // interpolate between cols
      ca = floor(clm - 0.5); cd = ca + 1; wc = (clm - 0.5) - ca;
    }

    if(zlm < 0.5){ // left part of first col
      za = 0; zd = za; wz = 0;
    }else if(zlm >= dimz-0.5){ // right part of last col
      za = dimz-1; zd = za; wz = 0;
    }else{ // interpolate between cols
      za = floor(zlm - 0.5); zd = za + 1; wz = (zlm - 0.5) - za;
    }

    out[i] =
      in[ra * nperrow + ca * dimz + za] * (1-wr)*(1-wc)*(1-wz)
      + in[ra * nperrow + ca * dimz + zd] * (1-wr)*(1-wc)*wz
      + in[ra * nperrow + cd * dimz + za] * (1-wr)*wc*(1-wz)
      + in[ra * nperrow + cd * dimz + zd] * (1-wr)*wc*wz
      + in[rd * nperrow + ca * dimz + za] * wr*(1-wc)*(1-wz)
      + in[rd * nperrow + ca*dimz + zd] * wr*(1-wc)*wz
      + in[rd * nperrow + cd*dimz + za] * wr*wc*(1-wz)
      + in[rd * nperrow + cd*dimz + zd] *wr*wc*wz;
  }
}
void cuda_3d_linear_interpolation(int n, float *g_in, float *g_out, float *g_dr, float* g_dc, float *g_dz, int dimr, int dimc, int dimz){
  kernel_3d_linear_interpolation<<<nblocks(n),blocksize()>>>(n, g_in, g_out, g_dr, g_dc, g_dz, dimc*dimz, dimr, dimc, dimz);
  cuda_error_handle();
}


__global__
void kernel_2d_linear_interpolation(int n, float *in, float *out, float *dr, float *dc, int dimr, int dimc){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n){

   // Scale coordinates up to N-by-M
   double rlm = dr[i] *  dimr;
   double clm = dc[i] * dimc;

   int ra, ca, rd, cd;
   float wr, wc;
   // Get surrounding pixel coordinates
   if(rlm < 0.5){ // upper part of first row
     ra = 0; rd = ra; wr = 0;
   }else if(rlm >= dimr-0.5){ // lower part of last row
     ra = dimr-1; rd = ra; wr = 0;
   }else{ // interpolate between rows
     ra = floor(rlm - 0.5); rd = ra + 1; wr = (rlm - 0.5) - ra;
   }

   if(clm < 0.5){ // left part of first col
     ca = 0; cd = ca; wc = 0;
   }else if(clm >= dimc-0.5){ // right part of last col
     ca = dimc-1; cd = ca; wc = 0;
   }else{ // interpolate between cols
     ca = floor(clm - 0.5); cd = ca + 1; wc = (clm - 0.5) - ca;
   }

   out[i] =
     in[ra * dimc + ca] * (1-wr) * (1-wc)
     + in[ra * dimc + cd] * (1-wr) * wc
     + in[rd * dimc + ca] * wr * (1-wc)
     + in[rd * dimc + cd] * wr * wc;

 }
}
void cuda_2d_linear_interpolation(int n, float *g_in, float *g_out, float *g_dr, float* g_dc, int dimr, int dimc){
  kernel_2d_linear_interpolation<<<nblocks(n),blocksize()>>>(n, g_in, g_out, g_dr, g_dc, dimr, dimc);
  cuda_error_handle();
}


/**
 * ==================================================================
 * Reductions
 * ==================================================================
 **/
/**
 * SUM
 * see https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
 **/
__global__
void kernel_sum(int n, float *g_in, float *g_out){

  extern __shared__ float sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  sdata[tid] = 0; // Initialize all thread sums to 0 - this avoid problems when summing over threads in the last block
  if(i >= n) return;
  sdata[tid] = g_in[i];// + g_in[i+blockDim.x];  
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
float cuda_sum(int n, float *g_a){

  float *g_b;
  cudaMalloc(&g_b, nblocks(n) * sizeof(float));
  cuda_error_handle();
  
  kernel_sum<<<nblocks(n),blocksize(), blocksize()  * sizeof(float)>>>(n, g_a, g_b);
  cuda_error_handle();
  
  float res2[nblocks(n)];  
  cudaMemcpy(&res2, g_b, nblocks(n) * sizeof(float), cudaMemcpyDeviceToHost);
  cuda_error_handle();
  
  float res = 0;
  for(int i=0; i<nblocks(n); i++){
    res += res2[i];
  }

  cudaFree(g_b);
  cuda_error_handle();
  return res;
}





__global__
void kernel_max(int n, float *g_in, float *g_out){

  extern __shared__ float sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  sdata[tid] = 0; // Initialize all thread sums to 0 - this avoid problems when summing over threads in the last block
  if(i >= n) return;
  sdata[tid] = g_in[i];// + g_in[i+blockDim.x];  
  __syncthreads();

 
  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] = max(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) g_out[blockIdx.x] = sdata[0];
  
  
}
float cuda_max(int n, float *g_a){

  float *g_b;
  cudaMalloc(&g_b, nblocks(n) * sizeof(float));
  cuda_error_handle();
  
  kernel_max<<<nblocks(n),blocksize(), blocksize()  * sizeof(float)>>>(n, g_a, g_b);
  cuda_error_handle();
  
  float res2[nblocks(n)];  
  cudaMemcpy(&res2, g_b, nblocks(n) * sizeof(float), cudaMemcpyDeviceToHost);
  cuda_error_handle();
  
  float res = 0;
  for(int i=0; i<nblocks(n); i++){
    res = max(res, res2[i]);
  }

  cudaFree(g_b);
  cuda_error_handle();
  return res;
}


__global__
void kernel_cuda_beta_image(int n, float* a, float* alpha, float* b, float beta){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;

  b[i] = a[i] * alpha[i] + (1-alpha[i]) * beta;
}

void cuda_beta_image(int n, float* a, float* alpha, float* b, float beta){

  kernel_cuda_beta_image<<<nblocks(n), blocksize()>>>(n, a, alpha, b, beta);
  cuda_error_handle();
  
}




__global__
void kernel_magnitude(int n, int ndims, float *v0, float *v1, float *v2, float* target){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;

  target[i] = sqrt(v0[i]*v0[i] + v1[i]*v1[i] + (ndims == 3 ? v2[i]*v2[i] : 0));
}

float cuda_maximal_magnitude(int n, int ndims, float *v0, float *v1, float *v2){

  float* target = cuda_f32_allocate(n, false);
  kernel_magnitude<<<nblocks(n), blocksize()>>>(n, ndims, v0, v1, v2, target);

  float res = cuda_max(n, target);

  cuda_f32_free(target);

  return res;

}

/**
 * COPY VALUES
 **/

void cuda_copy(int n, float *g_a, float *g_b){
  cudaMemcpy(g_b, g_a, n*sizeof(float), cudaMemcpyDeviceToDevice);
  cuda_error_handle();
}


__global__
void kernel_copy_slice(int n, float* from, float* to, int nperrow_dest, int nperrow_src, int h0, int w0, int d0, int h1, int w1, int d1, int r0, int c0, int z0){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i > n) return;

  int rdest = i / nperrow_dest;
  int rsrc = rdest + r0;

  int cdest = (i % nperrow_dest) / d1;
  int csrc = cdest + c0;

  int zdest = i % d1;
  int zsrc = zdest + z0;

  int j = rsrc * nperrow_src + csrc * d0 + zsrc;
  to[i] = from[j];


}
void cuda_copy_slice(int n, float *from, float *to, int h0, int w0, int d0, int h1, int w1, int d1, int r0, int c0, int z0){

  int nperrow_dest = d1 * w1;
  int nperrow_src = d0 * w0;

  kernel_copy_slice<<<nblocks(n), blocksize()>>>(n, from, to, nperrow_dest, nperrow_src, h0, w0, d0, h1, w1, d1, r0, c0, z0);
  cuda_error_handle();
}

