#include "MSE.hpp"


template<typename T> double MSE<T>::measure(const Context* context) const{
  const MSEContext *c = dynamic_cast<const MSEContext*>(context);

  Image tmp = c->_M - c->_Sinv;
  ImageUtils::multiply(tmp, tmp, tmp);
  return ImageUtils::sum(tmp);
}

template<typename T> vector<Image> MSE<T>::gradient(const Context* context) const{
  const MSEContext *c = dynamic_cast<const MSEContext*>(context);

  Image diff = c->_M - c->_Sinv;
  ImageUtils::scalar(diff, (T) -2, diff);
  vector<Image> grads = ImageUtils::gradients(c->_M);
  for(uint i=0; i<c->_M.dimensions(); i++){
    ImageUtils::multiply(diff, grads.at(i), grads.at(i));
  }
  return grads;
}

template<typename T> void MSE<T>::load(ConfigurationTree tree){
  (void) tree; // surpress unused parameter warning
}
