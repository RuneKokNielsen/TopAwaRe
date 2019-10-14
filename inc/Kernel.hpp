#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "Image.hpp"
#include "Node.hpp"

enum IdKernel { gaussian, cauchy_navier };


/**
 * Abstract class for smoothing kernels.
 */
class Kernel: Conf::Node {

public:

  /**
   * Returns a new image that is the result of smoothening
   * a given image by this kernel.
   * \param im The image to smoothen.
   **/
  virtual Image apply(const Image im) = 0;

  /**
   * Returns a new image that is an approximation of the
   * inverse action of the smoothening. I.e. if
   * A = K(B) then B ≈ K^{-1}(A)
   * \param im The image to unsmooth.
   **/
  virtual Image inverse(const Image im) = 0;

  /**
   * Initialize the kernel to a given image meta, in
   * particular storing constructs related to dimensions.
   **/
  virtual void init(const ImageMeta meta) = 0;


  /**
   * See Optimizer::decreaseRegularity
   **/
  virtual bool decreaseRegularity() = 0;

};







#endif
