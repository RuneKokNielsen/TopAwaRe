#include <CauchyNavierKernel.hpp>
#include <math.h>



template<typename T> void CauchyNavierKernel<T>::init(const ImageMeta metain){

  ImageMeta meta = metain;
  meta.dtype = cpx;
#ifdef CUDA_GPU
  meta.m = cpu;
#endif
  if(_isAInitialized) return;
  if(_initializing){
    while(_initializing);
    return;
  }
  _initializing = true;
  
  // Compute A-matrix coefficients (Cauchy Navier in Fourier domain)
  _A = Image(meta, true);
  
  int d = meta.dimensions;
  int* dlengths = new int[d];
  T* deltas = new T[d];
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
  
  
  float *pa = _A.ptr<float>();
#pragma omp parallel for  
  for(uint i=0; i<meta.ncells; i++){
    int i2 = i * 2;
    pa[i2+1] = 0;
    for(int di=0; di<d; di++){
      float val = (1.0 - cos(2.0 * M_PI *
			     ((i / dlengths[di]) % dims[di]) * deltas[di])) / (deltas[di] * deltas[di]);
      
      pa[i2] += val;
    }

    pa[i2] = (_gamma + 2 * _alpha * pa[i2]);
    pa[i2] = pa[i2] * pa[i2];

  }

#ifdef CUDA_GPU
  _A.toGPU();
#endif
  _initializing = false;
  _isAInitialized = true;

  delete[] dlengths;
  delete[] deltas;
  delete[] dims;
}
/**
 * @TODO: implement GPU-accelerated FFT
 **/
template<typename T> Image CauchyNavierKernel<T>::apply(const Image im){
  init(im.meta());

  Image g;
  Image G = ImageUtils::fft(im);
  ImageUtils::divide(G, _A, G);
  g = ImageUtils::ifft(G, im.type());

  return g;
}
template<typename T> Image CauchyNavierKernel<T>::inverse(const Image im){
  init(im.meta());
  
  Image G = ImageUtils::fft(im);
  ImageUtils::multiply(G, _A, G);
  Image g = ImageUtils::ifft(G, im.type());

  return g;
}


template<typename T> void CauchyNavierKernel<T>::load(ConfigurationTree conf){
  _ags = conf.get<vector<float>>("alpha_gamma", {0.01});
  updateAlphaGamma();
}
