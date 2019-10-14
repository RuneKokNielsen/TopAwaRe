#ifndef GAUSSIANKERNEL_HPP
#define GAUSSIANKERNEL_HPP

#include <Kernel.hpp>

/**
 * Implementation of Gaussian smoothing.
 **/
template<typename T>
class GaussianKernel : public Kernel {

private:

  vector<float> _sigmas;
  T _sigma;
  Image _A;
  Image _Ainv;
  bool _isAInitialized = false;

public:

  GaussianKernel(){};
  GaussianKernel(vector<float> sigmas) :_sigmas(sigmas) {
    updateSigma();
  };

  Image apply(const Image im);
  Image inverse(const Image im);
  void init(const ImageMeta meta);
  bool decreaseRegularity(){
    return updateSigma();
  }

  void load(ConfigurationTree conf);

private:

  bool updateSigma(){
    if(_sigmas.size() == 0) return false;
    _sigma = _sigmas.back();
    _sigmas.pop_back();
    _isAInitialized = false;
    return true;
  }

};

template class GaussianKernel<float>;

#endif
