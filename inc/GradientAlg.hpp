#ifndef GRADIENT_ALG_HPP
#define GRADIENT_ALG_HPP

#include <vector>
#include "Image.hpp"
#include "Measure.hpp"
#include "Interpolator.hpp"
#include "Kernel.hpp"


/**
 * Abstract for gradient algorithms; yields the gradient
 * of some measure context w.r.t. some transformation model.
 **/

class GradientAlg
{

public:

  struct Context{
    vector<Image> transformation;
    vector<Image> velocity;
    measure::Context *measureContext;
    Interpolator *interpolator;
  };

  virtual ~GradientAlg(){};

  /**
   * Computes the gradient.
   * \param meta The ImageMeta of the images.
   * \param measure The similarity measure to optimize.
   * \param context a Context object of relevant local values.
   **/
  virtual std::vector<Image> computeGradients(ImageMeta meta, const measure::Measure* measure, const Context context = {}) = 0;

  /**
   * See Optimizer::decreaseRegularity
   **/
  virtual bool decreaseRegularity() = 0;

  virtual float getRegularity(){ return 0; };

  virtual Kernel *getKernel(){ return NULL; };
};


#endif
