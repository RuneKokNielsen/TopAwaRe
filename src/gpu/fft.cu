#include "gpu.h"
#include "fft.h"


/**
 * FFT
 **/

/**
 * @TODO: current procedure requires dimensions to stay the same.
 * R2C and C2R procedures of CUFFT does not provide this so
 * currently we have to move stuff to/from complex space for
 * some overhead.
 */

int n_fft = -1;
cufftHandle plan_fft;
void cuda_fft(uint *dims, float *g_in, cufftComplex *g_out){


  uint n = dims[0] * dims[1] * dims[2];

  if(n_fft >= 0 && (uint) n_fft != n){
    cufftDestroy(plan_fft);
    n_fft = -1;
  }
  
  if(n_fft < 0){
    cufftResult_t res = cufftPlan3d(&plan_fft, dims[0], dims[1], dims[2], CUFFT_C2C);
    n_fft = n;
    if(res){
      fprintf(stderr, "CUFFT error: Plan creation failed %i", res);
      exit(-1);
    }
  }
    

    //cufftHandle plan;

  


  cufftComplex *g_in_cpx = (cufftComplex*) cuda_f32_allocate(n * 2, false);
  cuda_real_complex(n, g_in, g_in_cpx);

  
  /* Use the CUFFT plan to transform the signal in place. */
  cufftResult_t res = cufftExecC2C(plan_fft, g_in_cpx, g_out, CUFFT_FORWARD);
  if (res){
    fprintf(stderr, "CUFFT error: ExecC2C Forward failed %i", res);
    exit(-1);
  }
  
  cudaFree(g_in_cpx);
  cuda_error_handle();  
 
}

int n_ifft = -1;
cufftHandle plan_ifft;
void cuda_ifft(uint *dims, cufftComplex *g_in, float *g_out){


  int n = dims[0] * dims[1] * dims[2];

  if(n_ifft >= 0 && n != n_ifft){
    cufftDestroy(plan_ifft);
    n_ifft = -1;
  }

  if(n_ifft < 0){
    cufftResult_t res = cufftPlan3d(&plan_ifft, dims[0], dims[1], dims[2], CUFFT_C2C);
    n_ifft = n;
    if(res){
      fprintf(stderr, "CUFFT error: Plan creation failed %i", res);
      exit(-1);
    }
  }



  cufftComplex *g_out_cpx = (cufftComplex*) cuda_f32_allocate(n * 2, false);
  
  /* Use the CUFFT plan to transform the signal in place. */
  cufftResult_t res = cufftExecC2C(plan_ifft, g_in, g_out_cpx, CUFFT_INVERSE);
  if (res){
    fprintf(stderr, "CUFFT error: ExecC2C Inverse failed %i", res);
    exit(-1);
  }

  cuda_complex_real(n, g_out_cpx, g_out);
  cudaFree(g_out_cpx);
  cuda_error_handle();  
  
}



/**
 * FFT SHIFT
 **/

__global__
void kernel_fft_shift(int n, cufftComplex *in, cufftComplex *out, int nr, int nc, int nz, float cr, float cc, float cz, int dr, int dc, int dz){
  
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n){
    int r = i / (nc * nz);
    int c = (i % (nc * nz)) / nz;
    int z = (i % (nc * nz)) % nz;

    int r1 = r < cr ? r + dr : (r == cr ? cr : r - dr);
    int c1 = c < cc ? c + dc : (c == cc ? cc : c - dc);
    int z1 = z < cz ? z + dz : (z == cz ? cz : z - dz);

    int j = r1 * nc * nz + c1 * nz + z1;

    out[i] = in[j];
  }

}
void cuda_fft_shift(uint *dims, cufftComplex *g_in, cufftComplex *g_out){
  uint n = dims[0] * dims[1] * dims[2];
  float cr = (dims[0] - 1) / 2.0;
  float cc = (dims[1] - 1) / 2.0;
  float cz = (dims[2] - 1) / 2.0;
  int dr = floor(cr) + 1;
  int dc = floor(cc) + 1;
  int dz = floor(cz) + 1;
  kernel_fft_shift<<<nblocks(n),blocksize()>>>(n, g_in, g_out, dims[0], dims[1], dims[2], cr, cc, cz, dr, dc, dz);
  cuda_error_handle();  
}
