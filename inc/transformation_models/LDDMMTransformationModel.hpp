#ifndef LDDMM_TRANSFORMATION_MODEL_HPP
#define LDDMM_TRANSFORMATION_MODEL_HPP

#include <TransformationModel.hpp>
#include <GradientAlg.hpp>
#include <Interpolator.hpp>
#include <IntegrationScheme.hpp>
#include "TransConcat.hpp"
#include "Filters.hpp"

/**
 * Implementation of the LDDMM based on
 * "Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms" by Beg et al.
 **/
class LDDMMTransformationModel : public TransformationModel
{

public:

  LDDMMTransformationModel(){
    reset();
  };


  ~LDDMMTransformationModel(){
  }

  void step();
  void init(Image S, Image M, measure::Measure *measure, measure::Context *context);
  void load(ConfigurationTree conf);
  uint timesteps(){ return _timesteps; };

  void reset(){
    _baseTransIsIdentity = true;
    _level = -1;
  }

  vector<Image> getTrans(int t){
    /**
     * If we are at the first scale level we simply return the transformation at time t.
     * Otherwise we need to compose it with the transformation computed at previous
     * scale levels.
     **/
    return _baseTransIsIdentity ? _phi01.at(t) : TransConcat::concatTransOne(_interpolator, _baseTrans, _phi01.at(t));
  }
  vector<Image> getTransInv(int t){
    return _phi10.at(t);
  }

  vector<Image> getFull(){
    return getTrans(0);
  }

  vector<Image> getFullInverse(){
    /**
     * When getting the full inverse, if previous transformations were computed on
     * earlier scale levels, we need to compose these with the inverse computed on
     * the current scale level.
     **/
    vector<Image> fullTransInv;
    if(_baseTransIsIdentity){
      return _phi10.at(_timesteps-1);
    }
    for(uint d=0; d<_M.meta().dimensions; d++){
      // Operands in composition is reversed for inverse deformations
      fullTransInv.push_back(Image(_M.meta(), false));
      _interpolator->interpolate(_phi10.at(_timesteps-1).at(d), _baseTransInv, fullTransInv.at(d));      
    }
    return fullTransInv;
  }

  vector<Image> getDiffeomorphicTrans(){
    return getFull();
  }

  vector<Image> getDiffeomorphicInverse(){
    return getFullInverse();
  }

  void initTrans();

  bool decreaseRegularity(){
    return _gradientAlg->decreaseRegularity();
  }

  double measure(int t);

private:

  /**
   * Integrates the current velocity field estimations into transformations.
   **/
  void integrate();

  /**
   * Estimates the length of the geodesics based on the current velocity field
   * estimations.
   **/
  void computeLength();

  /**
   * Performs constant-speed reparameterization of the estimated velocity fields.
   **/
  void reparameterize();

  // Current scale level
  int _level;
  // Current image meta
  ImageMeta _meta;
  /**
   * The number of iterations between constant-speed reparameterizations.
   * 0 means no reparameterization is done.
   **/
  int _reparameterize;

  /**
   * If greater than 0, describes the smoothing factor of elastic regularization,
   * i.e. a Gaussian smoothing of the entire velocity fields after adding the
   * gradient step.
   **/
  float _elasticSigma;
  Image _elasticGauss;

  /**
   * Scaling of largest displacement vector magnitude in each gradient step as
   * measured in voxels. Typical values are around 0.2 - 0.4.
   **/
  float _gradScale;

  // The target image.
  Image _S;
  // The source image.
  Image _M;

  // Current forward transformation at current scale level.
  vector<vector<Image> > _phi01;
  // Current inverse transformation at current scale level.
  vector<vector<Image> > _phi10;
  // Current velocity field at current scale level.
  vector<vector<Image> > _v;

  /**
   * If yes, this is the first scale level. Otherwise, then the
   * forward transformations must be routinely composed with previous
   * transformations computed at earlier scale levels.
   **/
  bool _baseTransIsIdentity;
  /**
   * The aggregated forward transformation computed on earlier scale levels.
   **/
  vector<Image> _baseTrans;
  /**
   * The aggregated inverse transformation computed on earlier scale levels.
   **/
  vector<Image> _baseTransInv;

  // Cartesian mesh corresponding to identity transformation.
  vector<Image> _mesh;

  // The underlying gradient algorithm.
  GradientAlg *_gradientAlg;

  // The integration scheme to go from velocity fields to transformations.
  IntegrationScheme *_integrationScheme;

  // The number of timesteps.
  uint _timesteps;

  // The delta time between timesteps.
  double _dt;

  // The interpolator to use for e.g. gradient computations and composition.
  Interpolator *_interpolator;

  // Estimated length of geodesic
  double _length = 0;

  // Local step iteration counter
  int _i;
};















#endif
