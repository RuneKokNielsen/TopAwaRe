#ifndef TRANSFORMATION_MODEL_HPP
#define TRANSFORMATION_MODEL_HPP

#include "Image.hpp"
#include "Node.hpp"
#include "Measure.hpp"

using namespace std;


enum IdTransformationModel { lddmm, lddmm_singularities };

/**
 * Abstract class for transformation models.
 **/
class TransformationModel: Conf::Node
{

public:

  /**
   * Inits data structures to the given images
   */
  virtual void init(Image S, Image M, measure::Measure *measure, measure::Context *context) = 0;
  virtual void reset(){};
  virtual ~TransformationModel(){};

  /**
   * Computes and applies one gradient step.
   */
  virtual void step() = 0;

  /**
   * Tells which time index to compute the measure to be optimized.
   **/
  virtual int measureIndex(){ return 0; };

  /**
   * Returns the forward transformation at time t.
   * \param t Timestep
   **/
  virtual vector<Image> getTrans(int t) = 0;
  /**
   * Returns the inverse transformation at time t.
   * \param t Timestep
   **/
  virtual vector<Image> getTransInv(int t) = 0;

  /**
   * Returns the full forward transformation.
   **/
  virtual vector<Image> getFull() = 0;
  /**
   * Returns the full inverse transformation.
   **/
  virtual vector<Image> getFullInverse() = 0;

  /**
   * Get only diffeomorphic part of the forward transformation.
   **/
  virtual vector<Image> getDiffeomorphicTrans() = 0;
  /**
   * Get only diffeomorphic part of the inverse transformation.
   **/
  virtual vector<Image> getDiffeomorphicInverse() = 0;


  /**
   * Compute the model measure at given time.
   * \param t Timestep.
   **/
  virtual double measure(int t) = 0;

  /**
   * Return number of timesteps of the transformation.
   **/
  virtual uint timesteps(){ return 1; };

  /**
   * When the gradient becomes too small the optimizer is asked to decrease
   * the amount of regularization one step. If this is achieved, it returns
   * true and optimization is continued. If no more deregularization is
   * allowed false is returned and the optimization should stop.
   **/
  virtual bool decreaseRegularity() = 0;


protected:
  measure::Context *_context;
  measure::Measure *_measure;
  double _learningRate;
};



#endif
