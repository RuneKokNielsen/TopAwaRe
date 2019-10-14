#ifndef CC_ALPHA_HPP
#define CC_ALPHA_HPP

#include <Measure.hpp>
#include <CC.hpp>

using namespace measure;

/**
 * Context for computing alpha version of the CC measure.
 **/
struct CCalphaContext : CCContext, AlphaContext{
  CCalphaContext(uint timesteps, float beta, float w): Context(timesteps){
    _beta = beta;
    _w = w;
    for(uint i=0; i<timesteps; i++){
      _subs.push_back(new CCalphaContext(0, beta, _w));
    }};

  CCalphaContext* getContext(uint t){return dynamic_cast<CCalphaContext*>(_subs.at(t));}


  virtual void update(vector<Image> phi, vector<Image> phiInv){
    _interpolator->interpolate(_I, phi, _M);
    cuda_beta_image(_M.meta().ncells, _M.gptr(), alpha.gptr(), _M.gptr(), _beta);
    _interpolator->interpolate(_S, phiInv, _Sinv);

    updateWindows();
  }

  virtual void updateAlpha(vector<Image> phi){
    _interpolator->interpolate(_I, phi, _M);

    ImageUtils::betaImage(_M, alpha, _M, _beta);

    updateWindows();
  }

  virtual void init(Image S, Image I, Interpolator *interpolator){
    CCContext::init(S, I, interpolator);
  }

  AlphaContext* copy(){
    CCalphaContext* c = new CCalphaContext(1, _beta, _w);
    c->init(_S, _I, _interpolator);
    return c;
  }


  // Background value. See TopAwaRe paper.
  float _beta;


};

/**
 * Alpha version of CC measure. Uses the configured value beta as background
 * intensity to handle changes in the alpha channel.
 * See the papers linked at https://github.com/RuneKokNielsen/TopAwaRe
 **/
template<typename T>
class CCalpha: public CC<T>
{

public:

  double measure(const Context *context) const;
  vector<Image> gradient(const Context *context) const;

  CCalphaContext *baseContext(uint timesteps) const{
    return new CCalphaContext(timesteps, _beta, CC<T>::_w);
  }


  void load(ConfigurationTree tree);

  float _beta;

};

template class CCalpha<float>;
template class CCalpha<double>;

#endif
