#ifndef SEMI_LAGRANGIAN_HPP
#define SEMI_LAGRANGIAN_HPP

#include <IntegrationScheme.hpp>
#include <Interpolator.hpp>

/**
 * Semi Lagrangian integration scheme as described in
 * "Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms" by Beg et al.
 *
 * Works by approximating the mean velocity over the course from timestep t to t+1 before taking
 * the actual step.
 **/
class SemiLagrangian: public IntegrationScheme
{

public:

  SemiLagrangian(){};
  SemiLagrangian(Interpolator *interpolator, double dt): _interpolator(interpolator), _dt(dt){};

  void init(ImageMeta meta);
  void integrate(const vector<vector<Image> > f, vector<vector<Image> > g, bool forward);

private:

  ImageMeta _meta;
  vector<Image> _mesh;
  vector<Image> _qmesh;
  vector<Image> _alphabase;
  vector<Image> _alpha;
  Interpolator *_interpolator;

  uint SEMILAGRANGIAN_ALPHA_ITERATIONS = 3;
  double _dt;

  void prepareMesh(const vector<vector<Image> > f, int t, bool forward);

  void load(ConfigurationTree conf);

};
#endif
