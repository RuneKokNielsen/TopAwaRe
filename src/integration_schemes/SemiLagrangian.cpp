#include <SemiLagrangian.hpp>


void SemiLagrangian::init(ImageMeta meta){
  _meta = meta;
  _mesh = ImageUtils::meshgrid(_meta);
  _alphabase.clear();
  _alpha.clear();
  _qmesh.clear();
  for(uint d=0; d<_meta.dimensions;d++){
    Image a = Image(_meta, true);
    _alphabase.push_back(a);
    _alpha.push_back(a);
    _qmesh.push_back(a.clone());
  }
}


void SemiLagrangian::integrate(const vector<vector<Image> > f, vector<vector<Image> > g, bool forward){
  if(forward){
    for(int t=f.size()-2; t>=0; t--){
      prepareMesh(f, t, forward);
      for(uint d=0; d<_meta.dimensions; d++){
        _interpolator->interpolate(g.at(t + 1).at(d), _qmesh, g.at(t).at(d));
      }
    }
  }
  else{
    for(uint t=1; t<f.size(); t++){
      prepareMesh(f, t, forward);
      for(uint d=0; d<_meta.dimensions; d++){
	_interpolator->interpolate(g.at(t - 1).at(d), _qmesh, g.at(t).at(d));
      }
    }
  }
}

void SemiLagrangian::prepareMesh(const vector<vector<Image> > f, int t, bool forward){
  for(uint d=0; d<_meta.dimensions; d++){
    _alphabase.at(d).copy_to(_alpha.at(d));
  }
  for(uint i=0; i<SEMILAGRANGIAN_ALPHA_ITERATIONS; i++){
    for(uint d=0; d<_meta.dimensions; d++){
      ImageUtils::subtract(_mesh.at(d), _alpha.at(d), _qmesh.at(d));
    }
    for(uint d=0; d<_meta.dimensions; d++){
      _interpolator->interpolate(f.at(t).at(d), _qmesh, _alpha.at(d));
      ImageUtils::scalar(_alpha.at(d), _dt/2, _alpha.at(d));
    }
  }
  if(forward){
    for(uint d=0; d<_meta.dimensions; d++){
      ImageUtils::m_add(_mesh.at(d), _alpha.at(d), _qmesh.at(d));
    };
  }else{
    for(uint d=0; d<_meta.dimensions; d++){
      ImageUtils::subtract(_mesh.at(d), _alpha.at(d), _qmesh.at(d));
    }
  }
}

void SemiLagrangian::load(ConfigurationTree conf){
  conf.requireFields({"interpolator"});
  _dt = conf.get<float>("delta_t", 0.1);
  _interpolator = (Interpolator*) Conf::loadTree(conf, "interpolator");
}
