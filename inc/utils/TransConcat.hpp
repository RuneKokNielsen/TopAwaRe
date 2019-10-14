#ifndef TRANS_CONCAT_HPP
#define TRANS_CONCAT_HPP

#include "Image.hpp"
#include "Interpolator.hpp"

namespace TransConcat{


  void concatTransOne(Interpolator *interpolator, vector<Image> t0, vector<Image > t1, vector<Image> target);
  
  vector<Image>  concatTransOne(Interpolator *interpolator, vector<Image> t0, vector<Image > t1);
  
  vector<vector<Image> > concatTransMany(Interpolator *interpolator, vector<Image> t0, vector<vector<Image> > t1s);

}
#endif
