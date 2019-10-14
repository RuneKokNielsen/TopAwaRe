#include <CCalpha.hpp>
#include <omp.h>
#include "TransConcat.hpp"

template<typename T> double CCalpha<T>::measure(const Context *context) const{

  if(_beta > 0){
    return CC<T>::measure(context);
  }else{
    throw logic_error("Not implemented: eta mode CCalpha");
  }
  
}

template<typename T> vector<Image> CCalpha<T>::gradient(const Context *context) const{  


  if(_beta > 0){
    return CC<T>::gradient(context);
  }else{
    throw logic_error("Not implemented: eta mode CCalpha");
  }
}


template<typename T> void CCalpha<T>::load(ConfigurationTree conf){
  CC<T>::load(conf);
  _beta = conf.get<float>("beta", 0);
}


