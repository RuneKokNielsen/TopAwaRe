#ifndef INTERPOLATOR_HPP
#define INTERPOLATOR_HPP

#include "Image.hpp"
#include "Node.hpp"

enum IdInterpolator { linear };

/**
 * Abstract class for image interpolator algorithms.
 **/
class Interpolator : Conf::Node
{

public:

  /**
   * Interpolates the values from an image at the coordinates given by a mesh.
   * \param in The image to interpolate from.
   * \param mesh The interpolation mesh.
   * \param out The target image.
   **/
  virtual void interpolate(const Image in, const vector<Image> mesh, Image out) = 0;

};

#define INTERPOLATOR_BOILERPLATE(method) \
  class method ## Interpolator: public Interpolator{\
  public:								\
    void interpolate(const Image in, const vector<Image> mesh, Image out){ \
      switch(in.meta().dtype){						\
      case f32: _ ## method ## Interpolator<float>::interpolate(in, mesh, out); break; \
      case f64: _ ## method ## Interpolator<double>::interpolate(in, mesh, out);break; \
      case uint8: _ ## method ## Interpolator<uint8_t>::interpolate(in, mesh, out);break; \
      case cpx: throw logic_error("Not implemented: complex number interpolation!"); break; \
      }									\
    }									\
    void load(ConfigurationTree conf){(void) conf;} \
  };									\
  template class _ ## method ## Interpolator<double>;\
  template class _ ## method ## Interpolator<uint8_t>;\
  template class _ ## method ## Interpolator<float>;\

#endif
