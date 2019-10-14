#ifndef REGISTRATION_HPP
#define REGISTRATION_HPP

#include "TransformationModel.hpp"
#include "Interpolator.hpp"
#include "Measure.hpp"
#include "Node.hpp"

/**
 * Controls the registration of a pair of images based on the
 * execution tree and returns the result.
 **/
class Registration : Conf::Node{

public:

  Registration(){};

  /**
   * Executes the registration procedure on a pair of images.
   * \param S Target image.
   * \param I Source image.
   **/
  tuple<Image, Image, vector<Image>, vector<Image>, vector<Image>, vector<Image>, measure::Context* > apply(const Image S, const Image I);

  void load(ConfigurationTree conf);

private:

  /**
   * If not at the last scale level, go to the next level
   * and return true. Otherwise return false.
   **/
  bool updateScale();

  /**
   * Downscale images to current scale level.
   **/
  void computeScale();

  // Target image.
  Image _S;
  // Source image.
  Image _I;
  // Target image scaled to current scale.
  Image _Ss;
  // Source image scaled to current scale.
  Image _Ms;
  // Moving image transformed to target space at current scale.
  Image _Ms0;
  // Target image transformed to source space at current scale.
  Image _Ss1;

  // The transformation model.
  TransformationModel*_opt;
  // An interpolator.
  Interpolator *_interpolator;
  // The measure to optimize
  measure::Measure *_measure;
  measure::Context* _baseContext;

  /**
   * If greater than 0, display the moving image after each iteration
   * and wait for _display ms
   **/
  int _display;

  // Max number of iterations at current scale.
  int _maxits;
  // Max iterations for remaining scales-
  vector<int> _maxitss;

  // Current scale
  float _scale;
  // Remaining scales to optimize at.
  vector<float> _scales;

  /**
   * Minimum convergence rate. If the measure changes less than
   * this rate over 10 iterations then the optimization at current
   * scale is stopped early.
   **/
  float _minConvergenceRate;

  /**
   * If > 0, saves checkpoint images at every _checkpoint iteration.
   * May be useful to see how the optimization progresses.
   **/
  int _checkpoint;
  /**
   * Path to a file containing an image like the one to be saved.
   **/
  string _checkpointSrc;
  /**
   * Path to the directory to put checkpoint images in.
   **/
  string _checkpointDir;
};







#endif
