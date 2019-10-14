#ifndef CAUCHYNAVIERKERNEL_HPP
#define CAUCHYNAVIERKERNEL_HPP

#include <Kernel.hpp>

/**
 * Implementation of the Cauchy-Navier elastic kernel.
 * See "Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms" by Beg et al.
 **/
template<typename T>
class CauchyNavierKernel : public Kernel {

private:

  vector<float> _ags;
  T _alpha;
  T _gamma;
  Image _A;
  bool _isAInitialized = false;
  bool _initializing = false;

public:

  CauchyNavierKernel(){};
  CauchyNavierKernel(vector<float> ags) :_ags(ags) {
    updateAlphaGamma();
  };

  Image apply(const Image im);
  Image inverse(const Image im);
  void init(const ImageMeta meta);
  bool decreaseRegularity(){
    return updateAlphaGamma();
  }

  void load(ConfigurationTree conf);

private:

  bool updateAlphaGamma(){
    if(_ags.size() == 0) return false;
    float ag = _ags.back();
    _ags.pop_back();
    _gamma = 1/(1+ag);
    _alpha = 1 - _gamma;;
    _isAInitialized = false;
    return true;
  }

};

template class CauchyNavierKernel<double>;
template class CauchyNavierKernel<float>;

#endif
