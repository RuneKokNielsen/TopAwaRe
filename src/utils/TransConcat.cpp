
#include "TransConcat.hpp"




void  TransConcat::concatTransOne(Interpolator *interpolator, vector<Image> t0, vector<Image > t1, vector<Image> target){

  for(uint d=0; d<t0.at(0).meta().dimensions; d++){
    interpolator->interpolate(t0.at(d), t1, target.at(d));
  }
  
}

vector<Image>  TransConcat::concatTransOne(Interpolator *interpolator, vector<Image> t0, vector<Image > t1){

  vector<Image> res;
  for(uint d=0; d<t0.at(0).meta().dimensions; d++){
    res.push_back(Image(t1.at(0).meta(), false));
  }
  concatTransOne(interpolator, t0, t1, res);

  return res;
}


vector<vector<Image> > TransConcat::concatTransMany(Interpolator *interpolator, vector<Image> t0, vector<vector<Image> > t1s){
  
  vector<vector<Image> > res;
  for(uint t=0; t<t1s.size(); t++){
    res.push_back(concatTransOne(interpolator, t0, t1s.at(t)));
  }

  return res;
}


