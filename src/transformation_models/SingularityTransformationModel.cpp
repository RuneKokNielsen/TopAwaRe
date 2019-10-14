#include "SingularityTransformationModel.hpp"
#include <cmath>
#include "TransConcat.hpp"
#include "NiftiInt.hpp"
#include "System.hpp"
#include "RegistrationFrame.hpp"
#include "Filters.hpp"
#include "ControlPointSD.hpp"
#include "Global.hpp"

template <typename T> void SingularityTransformationModel<T>::init(Image S, Image M, Measure *measure, measure::Context *context){
  
 
  _meta = S.meta();
  _mesh = ImageUtils::meshgrid(_meta);
  _Walpha = ImageUtils::meshgrid(_meta);
  _measure = measure;  
  // _I is the base moving image. This reference is saved for
  // computing the gradient numerically
  _S = S;
  _I = M;
  _M = Image(_meta, false);

  _cdist = 0;
  for(uint d=0; d<_meta.dimensions; d++){
    _cdist += 1.0/(_meta.dims().at(d) * _meta.dims().at(d));
  }
  _cdist = sqrt(_cdist);
  
 
  _context = dynamic_cast<AlphaContext*>(context);
  if(_context == NULL){
    cout << "Invalid choice of measure: SingularityTransformationModel requires Alpha type Measure\n";
    exit(-1);
  }
  Image Ialpha = _I.clone();
  for(uint t=0; t<_context->_timesteps; t++){
    _context->getContext(t)->alpha = Image(_meta, true);
    ImageUtils::s_add<T>(_context->getContext(t)->alpha, (T) 1, _context->getContext(t)->alpha);
  }
  
  preparePsis(_meta);


  _i = -1;    
  _baseOptimizer->init(S, M, measure, context);
  

  updateWalpha();
  updateAlphas();
}


template<typename T>
void SingularityTransformationModel<T>::step(){
  
  _i++;
  if(_optProportionIncrease > 0 && _i % _optProportionIncrease == 0) _optProportion++;
  _baseOptimizer->step();
  
  if(_i==0) updateWalpha();
  if(_i%(_optProportion + 1) == _optProportion){
    
    /**
     * Numerically estimate gradient of singularity parameters
     **/
    float h = _cdist * _h;
    vector<float> pigrad;

    vector<Image> trans0 = _baseOptimizer->getTrans(0);
    SingularityEval *eval = new SingularityEval(this, trans0);
    vector<float> x;
    for(uint i=0 ;i<_singularities.size(); i++){
      vector<float> pvec = _singularities.at(i)->toOptVec();
      for(uint j=0; j<pvec.size(); j++){
        x.push_back(pvec.at(j));
      }
    }
    int j = 0;
    for(uint p=0; p<_singularities.size(); p++){
      vector<float> xp = x;
      float g;

      for(uint i=0; i<_singularities.at(p)->optsize(); i++){
        float hi = h;
        xp = x;
        xp.at(j+i) += hi;
        g = eval->eval(xp);
        xp.at(j+i) = max(.0f, xp.at(j+i) - 2 * hi);
        g = g - eval->eval(xp);
#ifdef NORMALIZED
        //pigrad.push_back(g / hi);
	  
#else
        if(i==0){
          pigrad.push_back(g);
          //pigrad.push_back(g < 0 ? max(-hi * 100 , g / hi) : min(hi * 100, g / hi));
        }else if(i==1){
          pigrad.push_back(g < 0 ? max(-0.1f, g/hi) : min(0.1f, g/h));
        }else{
          pigrad.push_back(g < 0 ? max(-hi , g / hi) : min(hi, g / hi));
        }

        cout << i << " -> " <<  g << "\n";
	
#endif
      }
      j += _singularities.at(p)->optsize();
    }

    /**
     * Update singularities
     **/
    float t0 = eval->eval();
    vector<float> x1 = pigrad;
    j = 0;
    for(uint i=0; i<_singularities.size(); i++){
      vector<float> sx;
      
      for(uint k=0; k<_singularities.at(i)->optsize(); k++){
        if(k == 0){     
          sx.push_back(min(_singularities.at(i)->getMaxPi(),
                           min(x.at(j+k) + _cdist, max(x.at(j+k) - _cdist,
                                                       x.at(j+k) - x1.at(j+k) * _a))));
        }else{
          sx.push_back(
                       min(x.at(j+k) + _cdist, max(x.at(j+k) - _cdist,
                                                   x.at(j+k) - x1.at(j+k) * _a)));
        }
	
        cout << x1.at(j+k) << " -> " << sx.at(k) << "\n";
      }
      _singularities.at(i)->update(sx);
      
      j+=_singularities.at(i)->optsize();
    }
    float t1 = eval->eval();    
    if(false){
      if(t1 > t0){
        _a = _a * 0.1;
      }else if(t1 < t0){
        _a = _a * 1.1;
      }
    }
    
    delete eval;
    /**
     * Update topology changing transformation
     */
    updateWalpha();


  }  
  // Update alpha fields and transformations
  updateAlphas();
}


template<typename T> vector<vector<Image> > SingularityTransformationModel<T>::concatTrans(vector<Image> Walpha, vector<vector<Image> > trans){
  return TransConcat::concatTransMany(_interpolator, Walpha, trans);
}


template<typename T> void SingularityTransformationModel<T>::updateAlphas(){
  for(int t=timesteps()-1; t>=0; t--){

    updateAlpha(_baseOptimizer->getTrans(t), t);

  }


}

template<typename T> void SingularityTransformationModel<T>::updateAlpha(vector<Image> phi01t, int t){


  computeAlpha(phi01t, _singularities, _context->getContext(t)->alpha);

  
}

template<typename T> void SingularityTransformationModel<T>::updateWalpha(){
  computeWalpha(_singularities, _Walpha);
}

template<typename T> void SingularityTransformationModel<T>::computeAlpha(vector<Image> phi01t, vector<Singularity<T>*> singularities, Image alpha){


  vector<Image> mesh;
  vector<Image> Walpha;
  T** X = new T*[_meta.dimensions];
  T** XW = new T*[_meta.dimensions];
  for(uint d=0; d<_meta.dimensions; d++){
    mesh.push_back(phi01t.at(d).clone());
    Walpha.push_back(phi01t.at(d).clone());
          
#ifdef CUDA_GPU
    X[d] = (T*) mesh.at(d).gptr();
    XW[d] = (T*) Walpha.at(d).gptr();
#else
    X[d] = (T*) mesh.at(d).ptr<T>();
    XW[d] = (T*) Walpha.at(d).ptr<T>();
#endif
  }
    
  T* palpha;
#ifdef CUDA_GPU
  palpha = (T*) alpha.gptr();
#else
  palpha = alpha.ptr<T>();
#endif
  

  ImageUtils::scalar(alpha, 0.0, alpha);
  ImageUtils::s_add<T>(alpha, (T) 1, alpha);
  
  for(uint i=0; i<singularities.size(); i++){
    Singularity<T> *s = singularities.at(i);

    for(uint d=0; d<_meta.dimensions; d++){
      if(i > 0) Walpha.at(d).copy_to(mesh.at(d));
    }

    if(s->getPi() == 0) continue;

    if(s->isCurved()){
      vector<Image> tmp = TransConcat::concatTransOne(_interpolator, dynamic_cast<CurvedSingularity*>(singularities.at(i))->_psi, mesh);
      for(uint d=0; d<_meta.dimensions; d++){
        tmp.at(d).copy_to(mesh.at(d));
      }
    }

    if(s->getPi() > 0) s->fillAlpha(palpha, X, _meta, _cdist * _alphaBoundarySmoothing);

    if(i<singularities.size()-1){

      for(uint d=0; d<_meta.dimensions; d++){
        mesh.at(d).copy_to(Walpha.at(d));
      }
      
      if(s->getPi() > 0) s->fillWalpha(XW, X, _meta);
      
      if(s->isCurved()){
        vector<Image> tmp = TransConcat::concatTransOne(_interpolator, dynamic_cast<CurvedSingularity*>(singularities.at(i))->_psiInv, Walpha);
        for(uint d=0; d<_meta.dimensions; d++){
          tmp.at(d).copy_to(Walpha.at(d));
        }
      }

    }

    
  }

  delete[] X;
  delete[] XW;
  
}

template<typename T> void SingularityTransformationModel<T>::computeWalpha(vector<Singularity<T>*> singularities, vector<Image> Walpha){

  vector<Image> mesh;
  T** X = new T*[_meta.dimensions];
  T** XW = new T*[_meta.dimensions];
  for(uint d=0; d<_meta.dimensions; d++){
    mesh.push_back(_mesh.at(d).clone());
    mesh.at(d).copy_to(Walpha.at(d));

#ifdef CUDA_GPU
    X[d] = (T*) mesh.at(d).gptr();
    XW[d] = (T*) Walpha.at(d).gptr();
#else
    X[d] = (T*) mesh.at(d).ptr<T>();
    XW[d] = (T*) Walpha.at(d).ptr<T>();
#endif
  }


  for(uint i=0; i<singularities.size(); i++){
    Singularity<T>* s = singularities.at(i);

    for(uint d=0; d<_meta.dimensions; d++){
      if(i > 0){
        Walpha.at(d).copy_to(mesh.at(d));
      }
    }

    if(s->isCurved()){
      vector<Image> tmp = TransConcat::concatTransOne(_interpolator, dynamic_cast<CurvedSingularity*>(s)->_psi, mesh);
      for(uint d=0; d<_meta.dimensions; d++){
        tmp.at(d).copy_to(mesh.at(d));
        tmp.at(d).copy_to(Walpha.at(d));
      }
    }

    if(s->getPi() > 0) s->fillWalpha(XW, X, _meta);

    if(s->isCurved()){
      vector<Image> tmp = TransConcat::concatTransOne(_interpolator, dynamic_cast<CurvedSingularity*>(s)->_psiInv, Walpha);
      for(uint d=0; d<_meta.dimensions; d++){
        tmp.at(d).copy_to(Walpha.at(d));
      }
    }

  }

  delete[] X;
  delete[] XW;
}

template<typename T> vector<Image> SingularityTransformationModel<T>::computeWalphaInv(vector<Singularity<T>*> singularities){


  vector<Image> WalphaInv;
  vector<Image> mesh;
  T** X = new T*[_meta.dimensions];
  T** XW = new T*[_meta.dimensions];
  for(uint d=0; d<_meta.dimensions; d++){
    mesh.push_back(_mesh.at(d).clone());
    WalphaInv.push_back(_mesh.at(d).clone());

#ifdef CUDA_GPU
    X[d] = (T*) mesh.at(d).gptr();
    XW[d] = (T*) WalphaInv.at(d).gptr();
#else
    X[d] = (T*) mesh.at(d).ptr<T>();
    XW[d] = (T*) WalphaInv.at(d).ptr<T>();
#endif
  }

  for(int i=singularities.size()-1; i>=0; i--){
    Singularity<T>* s = singularities.at(i);

    for(uint d=0; d<_meta.dimensions; d++){
      if((uint) i < singularities.size()-1) WalphaInv.at(d).copy_to(mesh.at(d));
    }

    if(s->isCurved()){
      vector<Image> tmp = TransConcat::concatTransOne(_interpolator, dynamic_cast<CurvedSingularity*>(singularities.at(i))->_psi, mesh);
      for(uint d=0; d<_meta.dimensions; d++){
        tmp.at(d).copy_to(mesh.at(d));
        tmp.at(d).copy_to(WalphaInv.at(d));
      }
    }

    if(s->getPi() > 0) s->fillWalphaInv(XW, X, _meta);

    if(s->isCurved()){
      vector<Image> tmp = TransConcat::concatTransOne(_interpolator, dynamic_cast<CurvedSingularity*>(singularities.at(i))->_psiInv, WalphaInv);
      for(uint d=0; d<_meta.dimensions; d++){
        tmp.at(d).copy_to(WalphaInv.at(d));
      }
    }

    
  }

  delete[] X;
  delete[] XW;
  return WalphaInv;  
}


template<typename T>
void SingularityTransformationModel<T>::preparePsis(ImageMeta meta){
  Image tmp = Image(meta);
  for(uint i=0; i<_singularities.size(); i++){
    CurvedSingularity* cs = dynamic_cast<CurvedSingularity*>(_singularities.at(i));
    if(cs != NULL){

      // If singularity is registered to cache transformations, check if they are
      // already present on disk
      if(cs->doCache()){
        System::prepareOutputDir(_curvesDir);
        if(ifstream(getCacheKey(cs->_cacheKey)).good()){
          // Cache exists - proceed to load transformations
          string ext = _meta.dimensions == 2 ? ".png" : ".nii.gz";
          for(uint j=0; j<tmp.dimensions(); j++){

            Image psi_d = RegistrationFrame().loadImage(getCacheFile(cs->_cacheKey, j, false) + ext);
            Image psi_d_inv = RegistrationFrame().loadImage(getCacheFile(cs->_cacheKey, j, true) + ext);
#ifdef CUDA_GPU
            psi_d.toGPU();
            psi_d_inv.toGPU();
#endif
            cs->_psi.push_back(psi_d);
            cs->_psiInv.push_back(psi_d_inv);	    
          }
          continue;
        }	
      }
      
      Measure* curveMeasure = new ControlPointSD(cs, meta);
      measure::Context* curveContext = curveMeasure->baseContext(_psiOptimizer->timesteps());
      _psiOptimizer->reset();
      _psiOptimizer->init(tmp, tmp, curveMeasure, curveContext);

      vector<Image> curveMesh = cs->curvedToMesh();
      vector<Image> lineMesh = cs->straightToMesh();
      Image tmpmesh = curveMesh.at(0).clone();
      int iteration = 0;
      while(cs->maxIterations() == 0 || iteration < cs->maxIterations()){
        iteration++;
        _psiOptimizer->step();
        cout << "Iteration " << iteration << "\n";

        // Moving the control points is a bit convoluted. First, the curve is converted to
        // a mesh of points in each dimension. These meshes are then used to interpolate
        // the displacements from the transformations. These displacements are finally
        // applied to the control points
        for(uint t=0; t<_psiOptimizer->timesteps(); t++){
          ControlPointSDContext *ct = dynamic_cast<ControlPointSDContext*>(curveContext->getContext(t));
	  
	  
          for(uint j=0; j<_meta.dimensions; j++){
            _interpolator->interpolate(_psiOptimizer->getTrans(t).at(j), curveMesh, tmpmesh);
            ct->_curve->updateCurvedCoords(tmpmesh, j);
            _interpolator->interpolate(_psiOptimizer->getTransInv(t).at(j), lineMesh, tmpmesh);
            ct->_curve->updateStraightCoords(tmpmesh, j);
          }
	
        }

        double e = _psiOptimizer->measure(0);
        cout << e << "\n";	
      }

      // Copy resulting transformations
      for(uint i=0; i<tmp.dimensions(); i++){	
        cs->_psi.push_back(_psiOptimizer->getTrans(0).at(i).clone());
        cs->_psiInv.push_back(_psiOptimizer->getTransInv(_psiOptimizer->timesteps() - 1).at(i).clone());	
      }

      // If registered to cache, save transformations
      if(cs->doCache()){
        std::ofstream flag(getCacheKey(cs->_cacheKey));
        flag.close();
        for(uint i=0; i<tmp.dimensions(); i++){
          Image tmp = _psiOptimizer->getTrans(0).at(i);
          tmp.toCPU();
          cout << "HELLO " << SOURCE_PATH << "\n";
          RegistrationFrame().writeImage(tmp, SOURCE_PATH, getCacheFile(cs->_cacheKey, i, false));

          tmp = _psiOptimizer->getTransInv(_psiOptimizer->timesteps() - 1).at(i);
          tmp.toCPU();
          RegistrationFrame().writeImage(tmp, SOURCE_PATH, getCacheFile(cs->_cacheKey, i, true));
        }
      }
      
    }     
  }
  if(_psiOptimizer != NULL){
    delete _psiOptimizer;
    _psiOptimizer = NULL;
  }
}


template<typename T>
void SingularityTransformationModel<T>::load(ConfigurationTree conf){
  
  conf.requireFields({"transformation_model", "interpolator", "singularities"});

  _baseOptimizer = (TransformationModel*) Conf::loadTree(conf, "transformation_model");
  _interpolator = (Interpolator*) Conf::loadTree(conf, "interpolator");

  if(conf.hasField("psi_transformation_model")){
    _psiOptimizer = (TransformationModel*) Conf::loadTree(conf, "psi_transformation_model");
    _conf = new ConfigurationTree(conf);
  }

  vector<ConfigurationTree> children = conf.getChildren("singularities");
  for(uint i=0; i<children.size(); i++){
    _singularities.push_back((Singularity<T>*) Conf::loadTree(children.at(i)));
  }

  _alphaBoundarySmoothing = conf.get<float>("alpha_boundary_smoothing", 1);
  
  _curvesDir = conf.get<string>("curves_dir", "curves");

  _h = conf.get<float>("h", 0.1);
  _a = conf.get<float>("a", 0.01);
  
  _optProportion = conf.get<int>("opt_proportion", 1);
  _optProportionIncrease = conf.get<int>("opt_proportion_increase", 0);
}

