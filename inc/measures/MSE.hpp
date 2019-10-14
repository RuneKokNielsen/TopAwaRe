#ifndef MSE_HPP
#define MSE_HPP

#include "Measure.hpp"

using namespace measure;

/**
 * Context for computing the mean squared error measure.
 **/
struct MSEContext : virtual Context{

  MSEContext(uint timesteps) : Context(timesteps) {
    for(uint i=0; i<timesteps; i++){
      _subs.push_back(new MSEContext(0));
    }
  }

  MSEContext() : Context(1){}

  virtual MSEContext* getContext(uint t){ return t==0 ? this : dynamic_cast<MSEContext*>(_subs.at(t)); }

  virtual void update(vector<Image> phi, vector<Image> phiInv){
    _interpolator->interpolate(_I, phi, _M);
    _interpolator->interpolate(_S, phiInv, _Sinv);
  }

  virtual void init(Image S, Image I, Interpolator *interpolator){
    Context::init(S, I, interpolator);

    _M = I.clone();
    _Sinv = S.clone();

  }

  Image _M;
  Image _Sinv;
};

/**
 * Mean squared error measure.
 **/
template<typename T>
class MSE : public Measure{

public:

  double measure(const Context* context) const;
  vector<Image> gradient(const Context* context) const;

  MSEContext* baseContext(uint timesteps) const{
    return new MSEContext(timesteps);
  }

  void load(ConfigurationTree tree);

};

template class MSE<float>;

#endif
