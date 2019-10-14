#include <CC.hpp>
#include <omp.h>
#include "TransConcat.hpp"

template<typename T> double CC<T>::measure(const Context *context) const{

  const CCContext *c = dynamic_cast<const CCContext*>(context);

  Image tmp = Image(c->_A.meta(), false);
  cuda_cc_measure(tmp.meta().ncells, c->_A.gptr(), c->_B.gptr(), c->_C.gptr(), tmp.gptr());
  return 1 - ImageUtils::sum(tmp) / tmp.meta().ncells;
}

template<typename T> vector<Image> CC<T>::gradient(const Context *context) const{

  const CCContext *c = dynamic_cast<const CCContext*>(context);


  Image diff = Image(c->_A.meta(), false);
  cuda_cc_grad(c->_A.meta().ncells, c->_Mbar.gptr(),  c->_Sinvbar.gptr(), c->_A.gptr(), c->_B.gptr(), c->_C.gptr(), diff.gptr());

  vector<Image> gradb;
  gradb = ImageUtils::gradients(c->_M);

  for(uint i=0; i<c->_Mbar.dimensions(); i++){
    ImageUtils::multiply(diff, gradb.at(i), gradb.at(i));
  }

  return gradb;
}


template<typename T> void CC<T>::load(ConfigurationTree tree){
  _combineMSE = tree.get<bool>("combine_mse", false);
  _w = tree.get<int>("w", 2);
}


