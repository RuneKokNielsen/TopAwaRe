#include "Filters.hpp"
#include "math.h"


Image filters::gaussianFilter(ImageMeta meta, float sigma){

#ifdef CUDA_GPU
  meta.m = cpu;
#endif
  
  meta.dtype = cpx;
  Image f = Image(meta, true);

   
  int d = meta.dimensions;
  int* dlengths = new int[d];
  double* deltas = new double[d];
  int* dims = new int[d];
  std::vector<int> vdims = meta.dims();
  for(int i=0; i<d; i++){
    dims[i] = vdims.at(i);
    deltas[i] = 1.0 / dims[i];
    if(i == 0){
      dlengths[d-1] = 1;
    }else{
      dlengths[d-1-i] = dlengths[d-i] * vdims.at(d-i);
    }
  }

  double a = 1.0 / (2 * sigma * sigma);
  double fraq = 2 * M_PI * sigma * sigma;
  double scale = (1.0) / sqrt(fraq);
  
  complex *p = f.ptr<complex>();
  
#pragma omp parallel for  
  for(uint i=0; i<meta.ncells; i++){
    double res = 1;
    p[i] = 1;
    double dist = 0;
    for(uint di=0; di<meta.dimensions; di++){
      double dist_di = ((i / dlengths[di]) % dims[di]) * deltas[di] - 0.5;
      dist += dist_di * dist_di;
      res *= scale * sqrt(M_PI/a)*exp(-(M_PI*M_PI)*(dist_di*dist_di)/a);
    }
    p[i] = (complex) exp((-dist * sigma*sigma)/2.0);
  }
  
#ifdef CUDA_GPU
  f.toGPU();
#endif

  delete[] dlengths;
  delete[] deltas;
  delete[] dims;
  
  return f;
}

