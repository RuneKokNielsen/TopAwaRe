#ifndef CC_HPP
#define CC_HPP

#include <Measure.hpp>

using namespace measure;

/**
 * A context in which to compute the normalized cross correlation
 * measure and gradient.
 **/
struct CCContext : virtual Context{
  CCContext(uint timesteps, int w) : Context(timesteps), _w(w) {
    for(uint i=0; i<timesteps; i++){
      _subs.push_back(new CCContext(0, w));
    }};

  CCContext(): Context(1){};
  virtual CCContext* getContext(uint t){return t==0?this:dynamic_cast<CCContext*>(_subs.at(t));}

  // Local moving image
  Image _M;
  // Local inverse target image
  Image _Sinv;
  // Local mean normalized moving image
  Image _Mbar;
  // Local mean normalized inverse target image
  Image _Sinvbar;
  // Local helper variable (non-normalized cross-correlation)
  Image _A;
  // Local helper variable (m^n-scaled variance of local moving image)
  Image _B;
  // Local helper variable (m^n-scaled varaince of local inverse target image)
  Image _C;
  // Radius of correlation window
  int _w;

  virtual void update(vector<Image> phi, vector<Image> phiInv){
    _interpolator->interpolate(_I, phi, _M);
    _interpolator->interpolate(_S, phiInv, _Sinv);

    updateWindows();
  }

  virtual void init(Image S, Image I, Interpolator *interpolator){
    Context::init(S, I, interpolator);

    _M = I.clone();
    _Sinv = S.clone();
    _Mbar = Image(I.meta(), false);
    _Sinvbar = Image(I.meta(), false);
    _A = Image(I.meta(), false);
    _B = Image(I.meta(), false);
    _C = Image(I.meta(), false);
    updateWindows();
  }

  void updateWindows(){
    #ifdef CUDA_GPU
    cuda_cc_preproc(_M.meta().ncells, _M.gptr(), _Sinv.gptr(), _A.gptr(), _B.gptr(), _C.gptr(), _Mbar.gptr(), _Sinvbar.gptr(), _M.meta().height, _M.meta().width, _M.meta().depth, _w);
    #endif
  }

};

/**
 * Normalized cross correlation measure as used in e.g.
 * "Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain." by Avants et al.
 **/
template<typename T>
class CC: public Measure
{

public:

  double measure(const Context *context) const;
  vector<Image> gradient(const Context *context) const;

  CCContext *baseContext(uint timesteps) const{
    return new CCContext(timesteps, _w);
  }

  void load(ConfigurationTree tree);

  bool _combineMSE;
  int _w;
};

template class CC<float>;
template class CC<double>;

#endif
