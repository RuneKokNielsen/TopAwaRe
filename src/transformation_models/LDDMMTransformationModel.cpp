#include <LDDMMTransformationModel.hpp>
#include "LDDMMGradient.hpp"

void LDDMMTransformationModel::init(Image S, Image M, measure::Measure *measure, measure::Context *context){
  _level++;

  _meta = S.meta();
  _S = S;
  _M = M;
  _context = context;
  _measure = measure;
  // Initialize the integration scheme
  _integrationScheme->init(_meta);
  _length = 0;
  _mesh = ImageUtils::meshgrid(_meta);

  /**
   * If this is not the first scale level, we must initialize the
   * transformation based on previously estimated transformations.
   **/
  if(_level > 0){
    initTrans();
  }

  /**
   * Reset velocity fields and transformations.
   **/
  _phi01.clear();
  _phi10.clear();
  _v.clear();
  for(uint t=0; t<_timesteps; t++){
    _phi01.push_back(vector<Image>());
    _phi10.push_back(vector<Image>());
    _v.push_back(vector<Image>());

    for(uint d=0; d<_meta.dimensions; d++){
      _phi01.at(t).push_back(_mesh.at(d).clone());
      _phi10.at(t).push_back(_mesh.at(d).clone());
      _v.at(t).push_back(Image(_meta, true));
    }

  }


  _i = -1;

  /**
   * If elastic smoothing is activated, initialize the filter.
   **/
  if(_elasticSigma > 0){
    _elasticGauss = filters::gaussianFilter(_meta, _elasticSigma);
  }

}

void LDDMMTransformationModel::initTrans(){
  if(_baseTransIsIdentity){
    /**
     * If this is the second scale level, the initial transformation is
     * taken directly from the transformation estimated at the first step.
     **/
    _baseTrans = TransConcat::concatTransOne(_interpolator, _phi01.at(0), _mesh);
    _baseTransInv = TransConcat::concatTransOne(_interpolator, _phi10.at(_timesteps - 1), _mesh);
    _baseTransIsIdentity = false;
  }else{
    /**
     * If this is the third or more scale level, we must compose the
     * transformation estimated at the previous step with all the ones
     * before that one.
     **/
    _baseTrans = TransConcat::concatTransOne(_interpolator, _baseTrans, _phi01.at(0));
    _baseTransInv = TransConcat::concatTransOne(_interpolator, _phi10.at(_timesteps-1), _baseTransInv);

    _baseTrans = TransConcat::concatTransOne(_interpolator, _baseTrans, _mesh);
    _baseTransInv = TransConcat::concatTransOne(_interpolator, _baseTransInv, _mesh);
  }
}

void LDDMMTransformationModel::step(){
  _i++;
  // Compute gradients and update velocity fields
  auto tt = TIC;

  for(uint t=0; t<_timesteps; t++){

    /**
     * Compute gradient at this time step.
     **/
    GradientAlg::Context context {_phi01.at(t), _v.at(t), _context->getContext(t), _interpolator};
    tt = TIC;
    vector<Image> grads = _gradientAlg->computeGradients(_meta, _measure, context);


    /**
     * Compute the maximal magnitude of any displacement in this gradient.
     * The gradient is then rescaled to some fixed maximal magnitude.
     **/
    float magmax1 = 1;
#ifdef CUDA_GPU
    if(_meta.dimensions == 2){
      magmax1 = cuda_maximal_magnitude(_meta.ncells, 2, grads.at(0).gptr(), grads.at(1).gptr(), NULL);
    }else{
      magmax1 = cuda_maximal_magnitude(_meta.ncells, 3, grads.at(0).gptr(), grads.at(1).gptr(), grads.at(2).gptr());
    }
#endif

    /**
     * Rescale gradient and update velocity field.
     **/
    for(uint d=0; d<_meta.dimensions; d++){
      ImageUtils::scalar(grads.at(d), _gradScale / magmax1 / _meta.dim(0),  grads.at(d));
      ImageUtils::subtract(_v.at(t).at(d), grads.at(d), _v.at(t).at(d));

      /**
       * If elastic regularization is activated, smooth the updated velocity field.
       **/
      if(_elasticSigma > 0){
        _v.at(t).at(d) = ImageUtils::filter(_v.at(t).at(d), _elasticGauss);
      }
    }
  }

  /**
   * Integrate the velocity fields into transformations.
   **/
  integrate();

  /**
   * Estimate the length of the geodesic.
   **/
  computeLength();


  /**
   * If activated, perform constant-speed reparameterization.
   **/
  if(_reparameterize > 0 && _i % _reparameterize == _reparameterize - 1){
    cout << "Reparameterize..\n";
    reparameterize();
    computeLength();
  }

}

void LDDMMTransformationModel::computeLength(){
  float sigma = _gradientAlg->getRegularity();
  if(sigma == 0) return;
  double l = 0;
  double dt = 1.0/_timesteps;
  for(uint t=0; t<_timesteps; t++){
    double tl = 0;
    for(uint d=0; d<_meta.dimensions; d++){
      Image Kinv = _gradientAlg->getKernel()->inverse(_v.at(t).at(d));
      ImageUtils::multiply(Kinv, Kinv, Kinv);
      tl += ImageUtils::sum(Kinv) / _v.at(t).at(d).meta().dim(d);;
    }
    l += dt * tl;
  };
}

void LDDMMTransformationModel::reparameterize(){
  double dt = 1.0/_timesteps;
  for(uint t=0; t<_timesteps; t++){
    double tl = 0;
    for(uint d=0; d<_meta.dimensions; d++){
      Image Kinv = _gradientAlg->getKernel()->inverse(_v.at(t).at(d));
      ImageUtils::multiply(Kinv, Kinv, Kinv);
      tl += ImageUtils::sum(Kinv);
    }
    tl *= dt;
    double mult = _length / ((double) _timesteps * tl);
    for(uint d=0; d<_meta.dimensions; d++){
      ImageUtils::scalar(_v.at(t).at(d), mult, _v.at(t).at(d));
    }
  }
}

void LDDMMTransformationModel::integrate(){
  _integrationScheme->integrate(_v, _phi01, true);
  _integrationScheme->integrate(_v, _phi10, false);
}


double LDDMMTransformationModel::measure(int t){
  float sigma = _gradientAlg->getRegularity();
  if(sigma > 0){
    return _measure->measure(_context->getContext(t)) + _length * sigma*sigma;
  }else{
    return _measure->measure(_context->getContext(t));
  }
}


void LDDMMTransformationModel::load(ConfigurationTree conf){

  conf.requireFields({"sigma", "kernel", "integration_scheme", "interpolator"});

  Kernel *k = (Kernel*) Conf::loadTree(conf, "kernel");
  _gradientAlg = new LDDMMGradient<float>(conf.get<vector<float>>("sigma"), k);

  _integrationScheme = (IntegrationScheme*) Conf::loadTree(conf, "integration_scheme");

  _timesteps = conf.get<int>("N", 10);
  _dt = conf.get<float>("T", 1) / _timesteps;


  _interpolator = (Interpolator*) Conf::loadTree(conf, "interpolator");

  _elasticSigma = conf.get<float>("elastic_sigma", 0);

  _reparameterize = conf.get<int>("reparameterize", 0);

  _gradScale = conf.get<float>("grad_scale", 0.2);
}
