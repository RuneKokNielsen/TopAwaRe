#include <LDDMMGradient.hpp>


template<typename T>std::vector<Image> LDDMMGradient<T>::computeGradients(ImageMeta meta, const measure::Measure* measure, const Context context){

  vector<Image> gradients = measure->gradient(context.measureContext);

  Image jac = ImageUtils::jacobianDeterminants(context.transformation);

  for(unsigned int i=0; i<meta.dimensions; i++){

    ImageUtils::multiply(jac, gradients.at(i), gradients.at(i));
    gradients.at(i) = _K->apply(gradients.at(i));
    Image tmp = Image(meta, true);
    if(_sigma > 0){
      ImageUtils::scalar(context.velocity.at(i), (T) 2*_sigma*_sigma / gradients.at(i).meta().dim(i), tmp);
    }
    ImageUtils::subtract(tmp, gradients.at(i), gradients.at(i));
  }

  return gradients;
}

