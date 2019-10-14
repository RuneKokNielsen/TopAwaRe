#include <Registration.hpp>
#include <unistd.h>
#include "TransConcat.hpp"
#include "Filters.hpp"
#include "NiftiInt.hpp"
#include "RegistrationFrame.hpp"
#include "System.hpp"
#include "CCalpha.hpp"

bool Registration::updateScale(){
  // Set the new scale. If no more scales are left we are done.
  if(_scales.size() == 0) return false;
  _scale = _scales.back(); _scales.pop_back();
  _maxits = _maxitss.back(); _maxitss.pop_back();

  // Rescale the images
  computeScale();

  // Initialize the optimizer to the rescaled images
  _opt->init(_Ss, _Ms, _measure, _baseContext);

  // Initialize the measure contexts to the rescaled images
  for(uint i=0; i<_opt->timesteps(); i++){
    _baseContext->getContext(i)->init(_Ss, _Ms, _interpolator);
    _baseContext->getContext(i)->update(_opt->getTrans(i), _opt->getTransInv(i));
  }

  // Update the transformed images to the new scale
  _Ms0 = _Ms.clone();
  _Ss1 = _Ss.clone();
  _interpolator->interpolate(_Ms, _opt->getTrans(0), _Ms0);
  _interpolator->interpolate(_Ss, _opt->getTrans(_opt->timesteps() - 1), _Ss1);

  return true;
}

void Registration::computeScale(){
  if(_scale < 1){
    /**
     * The scale is defined in terms of reduction such that
     * the size of a dimension with original size x becomes
     * x/(2^scale). A scale < 0 would mean an upscaling beyond
     * original size, so any scale < 1 is just handled as
     * the original size.
     **/
    _Ss = _S.clone();
    _Ms = _I.clone();
  }else{
    // Define dimensions of downscaled image
    float stride = pow(2, _scale);
    ImageMeta meta = _S.meta();
    meta.height = meta.height / stride;
    if(meta.height % 2 == 1) meta.height = meta.height + 1;
    meta.width = meta.width / stride;
    if(meta.width % 2 == 1) meta.width = meta.width + 1;
    meta.depth = meta.depth == 1 ? 1 : meta.depth / stride;
    if(meta.depth % 2 == 1 && meta.depth > 1) meta.depth = meta.depth + 1;

    _Ss = Image(meta, true);
    _Ms = Image(meta, true);

    // Filter original images by a Gaussian with sigma scale/2
    Image gauss = filters::gaussianFilter(_S.meta(), _scale / 2);
    Image Sg = ImageUtils::filter(_S, gauss);
    Image Mg = ImageUtils::filter(_I, gauss);

    // Interpolate downscaled image intensities from the smoothed images
    vector<Image> mesh = ImageUtils::meshgrid(_Ss.meta());
    _interpolator->interpolate(Sg, mesh, _Ss);
    _interpolator->interpolate(Mg, mesh, _Ms);
  }
}

tuple<Image, Image, vector<Image>, vector<Image>, vector<Image>, vector<Image>, measure::Context* > Registration::apply(const Image S, const Image I){

  _S = S;
  _I = I;
  _baseContext = _measure->baseContext(_opt->timesteps());

  updateScale();

  /**
   * The measureQueue keeps track of the past 10 measure values
   * in order to control early stopping.
   **/
  vector<double> measureQueue;
  int measureIndex = _opt->measureIndex();;
  double measure = _opt->measure(measureIndex);

  cout << "Initial distance: " << measure << endl;

  int i=-1;
  int j=-1;

  int skip = 0;
  TIC;
  bool newlevel = true;


  while(true){
    TOC;
    TIC;
    i++;
    j++;
    cout << "Iteration " << i << "\n";
    auto t = TIC;

    if(skip==0){
      // Take an optimization step
      _opt->step();
    }
    // Update images using full transformations
    _interpolator->interpolate(_Ms, _opt->getFull(), _Ms0);
    _interpolator->interpolate(_Ss, _opt->getFullInverse(), _Ss1);
    // Update similarity measure context based on new transformations
    for(uint t=0; t<_opt->timesteps(); t++){
      _baseContext->getContext(t)->update(_opt->getTrans(t), _opt->getTransInv(t));
    }
    tTOC(t);

    // Compute current measure
    measure = _opt->measure(measureIndex);
    cout << "Measure: " << measure << "\n";
    measureQueue.push_back(measure);
    if(measureQueue.size() > 10){
      measureQueue.erase(measureQueue.begin());
    }


    // Evaluate stopping conditions for current scale
    bool stop = false;
    skip = max(0, skip-1);
    if(0 < _maxits && _maxits <= j){
      // Stop if we exceeded the maximum number of iterations
      stop = true;
    }else if(measureQueue.size() == 10 && measure > measureQueue.at(0) * _minConvergenceRate){
      // Stop if we are not converging sufficiently
      stop = true;
    }
    if(stop){
      /**
       * If there are any more scales left, we go to next scale step.
       * Otherwise we are done.
       **/
      if(updateScale()){
        cout << "Reducing scale space\n";
        skip = 1;

        j = -1;

        cout << "Reducing regularity\n";
        _opt->decreaseRegularity();
        stop = false;
        measureQueue.clear();
        newlevel = true;
      }
    }

    if(stop){
      /**
       * Optimization is done. Format results and return.
       **/
      cout << "STOP\n";
      vector<Image> mesh = ImageUtils::meshgrid(I.meta());
      vector<Image> deformation = _opt->getFull();
      vector<Image> diffeo = _opt->getDiffeomorphicTrans();
      vector<Image> invdiffeo = _opt->getDiffeomorphicInverse();
      vector<Image> invfull = _opt->getFullInverse();
      Image S1 = Image(S.meta(), false);
      Image M1 = Image(I.meta(), false);

      _interpolator->interpolate(S, invfull, S1);
      _interpolator->interpolate(I, deformation, M1);
      for(uint d=0; d<S.meta().dimensions; d++){
        deformation.at(d) = deformation.at(d) - mesh.at(d);
        diffeo.at(d) = diffeo.at(d) - mesh.at(d);
        invdiffeo.at(d) = invdiffeo.at(d) - mesh.at(d);
        invfull.at(d) = invfull.at(d) - mesh.at(d);
      }
      if(_display>0){
        SHOWWAIT(M1, _display);
      }
      return make_tuple(M1, S1, deformation, invfull, diffeo, invdiffeo, _baseContext);
    }

    /**
     * If checkpointing is enabled, current image estimates are
     * routinely written to disk.
     **/
    if(_checkpoint > 0 && (i % _checkpoint == _checkpoint - 1 || newlevel)){
      newlevel = false;
      cout << "Write checkpoint..\n";
      string postfix = S.dimensions() == 3 ? ".nii.gz" : ".png";
      Image tmp = _Ms0;;
      CCalphaContext* alphacontext = dynamic_cast<CCalphaContext*>(_baseContext);
      if(alphacontext != NULL){
        ImageUtils::multiply(_Ms0, alphacontext->getContext(0)->alpha, tmp);
        tmp = alphacontext->getContext(0)->_I;
      }

      tmp.toCPU();
      RegistrationFrame().writeImage(tmp, _checkpointSrc, _checkpointDir + "/" + std::to_string(i) + "_M" + postfix);

      tmp = _Ss;
      tmp.toCPU();
      RegistrationFrame().writeImage(tmp, _checkpointSrc, _checkpointDir + "/" + std::to_string(i) + "_Sinv" + postfix);
    }

    /**
     * If displaying is activated, display the current
     * moving image at very iteration.
     **/
    if(_display > 0){

      measure::AlphaContext* alphacontext = dynamic_cast<measure::AlphaContext*>(_baseContext);
      if(alphacontext != NULL){
        ImageUtils::multiply(_Ms0, alphacontext->getContext(0)->alpha, _Ms0);
        Image alphainv = Image(_Ms0.meta(), true);
        ImageUtils::s_add(alphainv, 1.0, alphainv);
        ImageUtils::subtract(alphainv, alphacontext->getContext(0)->alpha, alphainv);
        Image tmp = _S * alphainv;
        ImageUtils::m_add(tmp, _Ms0, tmp);
        tmp.toCPU();
        RegistrationFrame().writeImage(tmp, "", "animation/frame-" + std::to_string(i));
      }else{
        Image tmp = _Ms0;
        tmp.toCPU();
        RegistrationFrame().writeImage(tmp, "", "animation/frame-" + std::to_string(i));
      }
      SHOWWAIT(_Ms0, _display);
    }
  }
}

void Registration::load(ConfigurationTree conf){

  conf.requireFields({"transformation_model", "interpolator", "measure", "maxits", "scale"});
  
  _opt = (TransformationModel*) Conf::loadTree(conf, "transformation_model");

  _interpolator = (Interpolator*) Conf::loadTree(conf, "interpolator");

  _measure = (measure::Measure*) Conf::loadTree(conf, "measure");

  _display = conf.get<int>("display", 0);

  _maxitss = conf.get<vector<int>>("maxits", {0});

  _scales = conf.get<vector<float>>("scale");

  _minConvergenceRate = conf.get<float>("min_convergence_rate", 0.999);

  _checkpoint = conf.get<int>("checkpoint", 0);
  _checkpointSrc = conf.get<string>("checkpoint_source", "");
  _checkpointDir = conf.get<string>("checkpoint_dir", "");
  if(_checkpointDir != ""){
    System::prepareOutputDir(_checkpointDir);
  }
}
