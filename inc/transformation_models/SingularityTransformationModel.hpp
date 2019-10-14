#ifndef SINGULARITY_TRANSFORMATION_MODEL_HPP
#define SINGULARITY_TRANSFORMATION_MODEL_HPP

#include "TransformationModel.hpp"
#include "Interpolator.hpp"
#include "Evaluator.hpp"
#include <iostream>
#include <fstream>
#include <algorithm> 
#include <cmath>
#include "TransConcat.hpp"
#include "Singularities.hpp"
#include <iostream>
#include <fstream>



using namespace measure;

/**
 * Transformation model that optimizes singularity expansions
 * and composes the resulting piecewise-diffeomorphism with the
 * diffeomorphism optimized by an underlying transformation model,
 * (e.g. the LDDMM).
 * See the papers at https://github.com/RuneKokNielsen/TopAwaRe
 **/
template<typename T>
class SingularityTransformationModel : public TransformationModel
{
  


public:

  SingularityTransformationModel(){};

  ~SingularityTransformationModel(){

  }

  void step();

  void load(ConfigurationTree conf);

  vector<Image> getTrans(int t){
    vector<Image> trans = _baseOptimizer->getTrans(t);
    return TransConcat::concatTransOne(_interpolator, _Walpha, trans);
  }
  vector<Image> getTransInv(int t){
    return _baseOptimizer->getTransInv(t);
  }

  vector<Image> getFull(){
    vector<Image> trans = _baseOptimizer->getFull();
    return TransConcat::concatTransOne(_interpolator, _Walpha, trans);
  }

  vector<Image> getFullInverse(){
    vector<Image> baseInv = _baseOptimizer->getFullInverse();
    vector<Image> WalphaInv = computeWalphaInv(_singularities);
    return TransConcat::concatTransOne(_interpolator, baseInv, WalphaInv);
  }

  vector<Image> getDiffeomorphicTrans(){
    return _baseOptimizer->getDiffeomorphicTrans();
  }

  vector<Image> getDiffeomorphicInverse(){
    return _baseOptimizer->getDiffeomorphicInverse();
  }


  void init(Image S, Image M, Measure *measure, measure::Context *context);

  uint timesteps(){
    return _baseOptimizer->timesteps();
  }

  void addSingularity(Singularity<T> *s){
    _singularities.push_back(s);
  }

  bool decreaseRegularity(){
    _i = -1;
    return _baseOptimizer->decreaseRegularity();
  }

  void writeMeta(string dir){
     ofstream f;
     f.open (dir + "/singularities");
     for(uint i=0; i<_singularities.size(); i++){
       f << *(_singularities.at(i)) << "\n";
     }
     f.close();
  }

  double measure(int t){
    return _baseOptimizer->measure(t);
  }

  class SingularityEval : public Evaluator<vector<float>, float>{

  public:

    SingularityEval(SingularityTransformationModel *parent, vector<Image> trans0): _parent(parent), _trans0(trans0){
    };


    /**
     * Takes a parameterization of the singularities and evaluates the measure
     */
    float eval(vector<float> x){
      measure::AlphaContext* context = _parent->_context->getContext(_parent->measureIndex())->copy();
      vector<Singularity<T>*> sings = _parent->copySingularities();
      // If the input vector is empty, evaluate current parameters
      if(x.size() > 0){
        int j = 0;        
        for(uint i=0; i<sings.size(); i++){
          vector<float> sx;
          for(uint k=0; k<sings.at(i)->optsize(); k++){
            sx.push_back(x.at(j + k));
          }
          sings.at(i)->update(sx);
          j += sings.at(i)->optsize();
        }
      }

      context->alpha = Image(_parent->_S.meta());
      _parent->computeAlpha(_trans0, sings, context->alpha);
      vector<Image> Walpha;
      for(uint d=0; d<_parent->_meta.dimensions; d++){
        Walpha.push_back(_parent->_mesh.at(d).clone());
      }
      _parent->computeWalpha(sings, Walpha);
      for(int i=sings.size()-1; i>=0; i--){
        delete sings.at(i);
      }

      vector<Image>  phi01 = TransConcat::concatTransOne(_parent->_interpolator, Walpha, _trans0);
      context->updateAlpha(phi01);

      double m = _parent->_measure->measure(context);
      delete(context);
      return m;
    }

    float eval(){
      return eval(vector<float>());
    }

  private:
    SingularityTransformationModel *_parent;
    vector<Image> _trans0;
  };

private:


  vector<Singularity<T>*> _singularities;

  ImageMeta _meta;
  vector<Image> _mesh;

  measure::AlphaContext *_context;


  Image _I;
  Image _M;
  Image _S;
  TransformationModel *_baseOptimizer;
  Interpolator *_interpolator;
  Measure *_measure;
  TransformationModel *_psiOptimizer;

  void updateAlpha(vector<Image> phi01t, int t);
  void updateAlphas();
  void updateWalpha();
  void computeAlpha(vector<Image> phi01t, vector<Singularity<T>*> singularities, Image alpha);
  void preparePsis(ImageMeta meta);

  void computeWalpha(vector<Singularity<T>*> singularities, vector<Image> Walpha);
  vector<Image> computeWalphaInv(vector<Singularity<T>*> singularities);

  vector<vector<Image> > concatTrans(vector<Image> Walpha, vector<vector<Image> > baseTrans);

  vector<Image> _psi;
  vector<Image> _psiInv;

  float _cdist;
  float _alphaBoundarySmoothing;

  float _a;
  float _h;
  int _i;
  int _optProportion;
  int _optProportionIncrease;
  vector<Image> _Walpha;
  Image _sWalpha;
  vector<Image> _transInv;

  string _curvesDir;

  ConfigurationTree* _conf;

  string getCacheKey(string key){
    return _curvesDir + "/" + key;
  }

  string getCacheFile(string key, int dim, bool inv = false){
    return getCacheKey(key)  + (inv ? "_inv" : "") + "_" + std::to_string(dim);
  }

  int measureIndex(){
    return _baseOptimizer->measureIndex();
  }

  vector<Singularity<T>*> copySingularities(){
    vector<Singularity<T>*> sings;
    std::transform(_singularities.begin(), _singularities.end(), std::back_inserter(sings), std::mem_fun(&Singularity<T>::clone));
    return sings;
  }
};

template class SingularityTransformationModel<float>;
template class Singularity<float>;



#endif
