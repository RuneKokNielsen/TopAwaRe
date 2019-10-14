#include <GaussianKernel.hpp>
#include <math.h>
#include "Filters.hpp"


template<typename T> void GaussianKernel<T>::init(const ImageMeta metain){

  ImageMeta meta = metain;
  if(_isAInitialized) return;
  _isAInitialized = true;

  _A = filters::gaussianFilter(meta, _sigma);
  /*
  _Ainv = Image(_A.meta(), true);
  _Ainv = 1 + _Ainv;
  ImageUtils::divide(_Ainv, _A, _Ainv);
  */
}
/**
 * @TODO: implement GPU-accelerated FFT
 **/
template<typename T> Image GaussianKernel<T>::apply(const Image im){
  init(im.meta());
  return ImageUtils::filter(im, _A);

}
template<typename T> Image GaussianKernel<T>::inverse(const Image im){
  init(im.meta());  
  return ImageUtils::filter(im, _A);
  
}


template<typename T>
void GaussianKernel<T>::load(ConfigurationTree conf){
  _sigmas = conf.get<vector<float>>("sigma", {1});
  updateSigma();;
}
