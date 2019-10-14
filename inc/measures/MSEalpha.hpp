#ifndef MSE_ALPHA_HPP
#define MSE_ALPHA_HPP

#include "MSE.hpp"

using namespace measure;

/**
 * Context for computing the alpha version of the MSE.
 **/
struct MSEalphaContext : MSEContext, AlphaContext{

  MSEalphaContext(uint timesteps): Context(timesteps){
    for(uint i=0; i<timesteps; i++){
      _subs.push_back(new MSEalphaContext(0));
    }
  };

  MSEalphaContext* getContext(uint t){ return dynamic_cast<MSEalphaContext*>(_subs.at(t)); }

  virtual void update(vector<Image> phi, vector<Image> phiInv){
    MSEContext::update(phi, phiInv);
  }

  virtual void updateAlpha(vector<Image> phi){
    _interpolator->interpolate(_I, phi, _M);
  }

  virtual void init(Image S, Image I, Interpolator* interpolator){
    MSEContext::init(S, I, interpolator);
  }

  AlphaContext* copy(){
    MSEalphaContext* c = new MSEalphaContext(1);
    c->init(_S, _I, _interpolator);
    return c;
  }

};


/**
 * Alpha version of the MSE measure. Assumes topological holes to
 * match the target perfectly, but has a linear regularization
 * on the amount of background matter which is scaled by the
 * eta parameter.
 * See the papers linked at https://github.com/RuneKokNielsen/TopAwaRe
 **/
template<typename T>
class MSEalpha : public MSE<T>{

public:

  double measure(const Context* context) const;
  vector<Image> gradient(const Context* context) const;

  MSEalphaContext* baseContext(uint timesteps) const{
    return new MSEalphaContext(timesteps);
  }

  void load(ConfigurationTree tree);

private:

  float _eta;
};

template class MSEalpha<float>;

#endif
