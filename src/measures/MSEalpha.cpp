
#include "MSEalpha.hpp"

template<typename T> double MSEalpha<T>::measure(const Context* context) const{
  const MSEalphaContext* c = dynamic_cast<const MSEalphaContext*>(context);

  return cuda_mse_alpha_sum(c->_M.meta().ncells, _eta, c->_Sinv.gptr(), c->_M.gptr(), c->alpha.gptr());
}

template<typename T> vector<Image> MSEalpha<T>::gradient(const Context* context) const{
  const MSEalphaContext* c = dynamic_cast<const MSEalphaContext*>(context);

  Image diff = c->_M - c->_Sinv;
  Image diffl = c->alpha * diff * diff;
  Image a2 =  c->alpha * c->alpha;

  ImageUtils::scalar(diff, (T) -2, diff);
  ImageUtils::scalar(diffl, (T) -2, diffl);

  vector<Image> gradM = ImageUtils::gradients(c->_M);
  vector<Image> gradalpha = ImageUtils::gradients(c->alpha);

  for(uint i=0; i<c->_M.dimensions(); i++){
    ImageUtils::multiply(diff, gradM.at(i), gradM.at(i));
    ImageUtils::multiply(gradM.at(i), a2, gradM.at(i));

    ImageUtils::m_add(gradM.at(i), _eta * gradalpha.at(i), gradM.at(i));

    ImageUtils::multiply(diffl, gradalpha.at(i), gradalpha.at(i));

    ImageUtils::m_add(gradM.at(i), gradalpha.at(i), gradM.at(i));
  }

  return gradM;

}

template<typename T> void MSEalpha<T>::load(ConfigurationTree tree){
  _eta = tree.get<float>("eta", 0);
}
