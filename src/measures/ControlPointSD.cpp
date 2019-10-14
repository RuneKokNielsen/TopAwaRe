#include "ControlPointSD.hpp"
#include "math.h"
#include "Filters.hpp"

double ControlPointSD::measure(const Context *context) const{

  const ControlPointSDContext* c = static_cast<const ControlPointSDContext*>(context);


  CurvedSingularity* curve = c->_curve;
  float ssd = 0;
  for(uint i=0; i<curve->nCoords(); i++){
    for(uint j=0; j<3; j++){
      ssd += pow(curve->getCurvedCoord(i, j) - curve->getStraightCoord(i, j), 2);      
    }
  }
  

  return ssd;
}


vector<Image> ControlPointSD::gradient(const Context *context) const{

  const ControlPointSDContext* c = static_cast<const ControlPointSDContext*>(context);

  ImageMeta meta = _meta;
#ifdef CUDA_GPU
  meta.m = cpu;
#endif
  
  vector<Image> grads;
  int deltas[3] = {1};
  for(uint i=0; i<meta.dimensions; i++){
    grads.push_back(Image(meta, true));
    for(uint j=i+1; j<3; j++){
      deltas[i] *= meta.dim(j);
    }
  }

  CurvedSingularity* curve = c->_curve;
  #pragma omp parallel
  for(uint i=0; i<curve->nCoords(); i++){
    int x0[3] = {0};
    float dx0[3];
    float gradi[3];
    for(uint j=0; j<3; j++){

      x0[j] = floor(curve->getCurvedCoord(i, j) * meta.dim(j));
      dx0[j] = curve->getCurvedCoord(i, j) * meta.dim(j) - x0[j];
      gradi[j] = 2 * (curve->getStraightCoord(i, j) - curve->getCurvedCoord(i, j));
    }
    for(uint j=0; j<meta.dimensions; j++){
      grads.at(j).set<float>(x0[0], x0[1], x0[2], grads.at(j).get<float>(x0[0], x0[1], x0[2]) + gradi[j] * (1-dx0[0])*(1-dx0[1])*(1-dx0[2]));
      grads.at(j).set<float>(x0[0]+1, x0[1], x0[2], grads.at(j).get<float>(x0[0]+1, x0[1], x0[2]) + gradi[j] * (dx0[0])*(1-dx0[1])*(1-dx0[2]));
      grads.at(j).set<float>(x0[0], x0[1]+1, x0[2], grads.at(j).get<float>(x0[0], x0[1]+1, x0[2]) + gradi[j] * (1-dx0[0])*(dx0[1])*(1-dx0[2]));
      grads.at(j).set<float>(x0[0]+1, x0[1]+1, x0[2], grads.at(j).get<float>(x0[0]+1, x0[1]+1, x0[2]) + gradi[j] * (dx0[0])*(dx0[1])*(1-dx0[2]));

      if(meta.dimensions==3){
	grads.at(j).set<float>(x0[0], x0[1], x0[2]+1, grads.at(j).get<float>(x0[0], x0[1], x0[2]+1) + gradi[j] * (1-dx0[0])*(1-dx0[1])*(dx0[2]));
	grads.at(j).set<float>(x0[0]+1, x0[1], x0[2]+1, grads.at(j).get<float>(x0[0]+1, x0[1], x0[2]+1) + gradi[j] * (dx0[0])*(1-dx0[1])*(dx0[2]));
	grads.at(j).set<float>(x0[0], x0[1]+1, x0[2]+1, grads.at(j).get<float>(x0[0], x0[1]+1, x0[2]+1) + gradi[j] * (1-dx0[0])*(dx0[1])*(dx0[2]));
	grads.at(j).set<float>(x0[0]+1, x0[1]+1, x0[2]+1, grads.at(j).get<float>(x0[0]+1, x0[1]+1, x0[2]+1) + gradi[j] * (dx0[0])*(dx0[1])*(dx0[2]));
      }
    }
    
  }


#ifdef CUDA_GPU
  for(uint i=0; i<meta.dimensions; i++){
    grads.at(i).toGPU();
  }
#endif

  
  return grads;
}



void ControlPointSD::load(ConfigurationTree conf){
  (void) conf;
}

