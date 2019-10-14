#ifndef FILTERS_HPP
#define FILTERS_HPP

#include "Image.hpp"

namespace filters{


  Image gaussianFilter(ImageMeta meta, float sigma);

  Image discreteGaussianFilter(int width, float sigma);
}







#endif
