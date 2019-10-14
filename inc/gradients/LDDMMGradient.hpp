#ifndef LDDMM_GRADIENT_HPP
#define LDDMM_GRADIENT_HPP

#include <GradientAlg.hpp>
#include <Kernel.hpp>
/**
 * Beg LDDMM gradient implementation with free kernel and measure choice.
 * See "Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms" by Beg et al.
 */
template<typename T>
class LDDMMGradient : public GradientAlg
{


public:

  LDDMMGradient(vector<float> sigmas, Kernel *K): _sigmas(sigmas), _K(K){
    updateSigma();
  };

  std::vector<Image> computeGradients(ImageMeta meta, const measure::Measure* measure, const Context context = {});


  bool decreaseRegularity(){
    updateSigma();
    return _K->decreaseRegularity();
  }

  float getRegularity(){
    return _sigma;
  }

  Kernel *getKernel(){
    return _K;
  }
private:

  /**
   * Vector of sigma values for each scale step.
   **/
  vector<float> _sigmas;

  /**
   * Current smoothing sigma. Higher values increase regularity.
   **/
  T _sigma;
  Kernel *_K;

  void updateSigma(){
    _sigma = _sigmas.back();
    _sigmas.pop_back();
  }

};

template class LDDMMGradient<float>;
template class LDDMMGradient<double>;

#endif
