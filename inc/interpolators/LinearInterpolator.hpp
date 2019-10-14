#ifndef LINEAR_INTERPOLATOR_HPP
#define LINEAR_INTERPOLATOR_HPP

#include <Interpolator.hpp>


/**
 * Linear image interpolator.
 **/
template<typename T>
class _LinearInterpolator
{
public:
  static void interpolate(const Image in, const vector<Image> mesh, Image out);

};

INTERPOLATOR_BOILERPLATE(Linear)





#endif
