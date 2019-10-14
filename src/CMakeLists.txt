
add_library(src Image.cpp Registration.cpp RegistrationFrame.cpp
  configuration/ConfigurationTree.cpp configuration/Node.cpp
  gradients/LDDMMGradient.cpp
  integration_schemes/SemiLagrangian.cpp
  interpolators/LinearInterpolator.cpp
  kernels/CauchyNavierKernel.cpp  kernels/GaussianKernel.cpp
  measures/ControlPointSD.cpp measures/CC.cpp measures/CCalpha.cpp measures/MSE.cpp measures/MSEalpha.cpp
  transformation_models/LDDMMTransformationModel.cpp transformation_models/SingularityTransformationModel.cpp transformation_models/Singularities.cpp
  utils/Filters.cpp utils/TransConcat.cpp utils/Global.cpp)

if(with_gpu)
  target_link_libraries(src libgpu)
endif()

target_link_libraries(src)
