#ifndef CONTROLPOINTSD_HPP
#define CONTROLPOINTSD_HPP


#include "Measure.hpp"
#include "Singularities.hpp"
using namespace measure;

struct ControlPointSDContext : Context{



  
  ControlPointSDContext(uint timesteps, CurvedSingularity* curve) : Context(timesteps), _curve(curve){
    subs.push_back(this);
    for(uint i=1; i<timesteps; i++){
      subs.push_back(new ControlPointSDContext(1, curve->clone()));
    }
  }


  string name(){ return "ControlPointSD"; }
  ControlPointSDContext* getContext(uint t){ return t==0?this:subs.at(t); }

  vector<ControlPointSDContext*> subs;
  CurvedSingularity* _curve;

};


class ControlPointSD : public Measure
{

public:

  CurvedSingularity* _curve;
  ImageMeta _meta;
  
  ControlPointSD(CurvedSingularity* curve, ImageMeta meta):_curve(curve),_meta(meta){};

  double measure(const Context* context) const;

  vector<Image> gradient(const Context* context) const; 
 

  ControlPointSDContext* baseContext(uint timesteps) const{
    return new ControlPointSDContext(timesteps, _curve);
  }

  void load(ConfigurationTree conf);

};


#endif
