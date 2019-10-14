#ifndef NODE_MAPPER_HPP
#define NODE_MAPPER_HPP

/**
 * \file NodeMapper.hpp
 * Maps configuration class names to new instances of corresponding objects.
 **/

#include "Node.hpp"
#include <map>
#include <boost/assign/list_of.hpp>

#include "CC.hpp"
#include "MSE.hpp"
#include "CCalpha.hpp"
#include "MSEalpha.hpp"
#include "GaussianKernel.hpp"
#include "CauchyNavierKernel.hpp"
#include "LinearInterpolator.hpp"
#include "SemiLagrangian.hpp"
#include "LDDMMTransformationModel.hpp"
#include "SingularityTransformationModel.hpp"
#include "Registration.hpp"
#include "RegistrationFrame.hpp"
#include "Singularities.hpp"
namespace NodeMapper{

  /**
   * Defines all configuration class names.
   **/
  enum NodeClass{
    /**
     * Metrics
     **/
    cc,
    cc_alpha,
    mse,
    mse_alpha,

    /**
     * Kernels
     **/
    gaussian,
    cauchy_navier,

    /**
     * Interpolators
     **/
    linear,

    /**
     * Intergration schemes
     **/
    semi_lagrangian,

    /**
     * Transformation models
     **/
    lddmm,
    singularity,

    /**
     * Framework
     **/
    registration,
    registration_frame,

    /**
     * Singularities
     **/
    point_singularity,
    line_singularity,
    curve_singularity,
    quadrilateral_singularity,
    curved_quadrilateral_singularity
  };

  /**
   * Maps a configuration class name string to corresponding enum.
   **/
  map<string, NodeClass> nodeClassMap = boost::assign::map_list_of
    ("cc", cc)
    ("cc_alpha", cc_alpha)
    ("mse", mse)
    ("mse_alpha", mse_alpha)
    ("gaussian", gaussian)
    ("cauchy_navier", cauchy_navier)
    ("linear", linear)
    ("semi_lagrangian", semi_lagrangian)
    ("lddmm", lddmm)
    ("singularity", singularity)
    ("registration", registration)
    ("registration_frame", registration_frame)
    ("point_singularity", point_singularity)
    ("line_singularity", line_singularity)
    ("curve_singularity", curve_singularity)
    ("quadrilateral_singularity", quadrilateral_singularity)
    ("curved_quadrilateral_singularity", curved_quadrilateral_singularity)
    ;

  /**
   * Creates new node instance by configuration class name.
   **/
  Conf::Node* loadNode(string str){

    try{
      switch(nodeClassMap.at(str)){
      case NodeClass::cc:
        return (Conf::Node*) new CC<float>();
      case NodeClass::cc_alpha:
        return (Conf::Node*) new CCalpha<float>();
      case NodeClass::mse:
        return (Conf::Node*) new MSE<float>();
      case NodeClass::mse_alpha:
        return (Conf::Node*) new MSEalpha<float>();
      case NodeClass::gaussian:
        return (Conf::Node*) new GaussianKernel<float>();
      case NodeClass::cauchy_navier:
        return (Conf::Node*) new CauchyNavierKernel<float>();
      case NodeClass::linear:
        return (Conf::Node*) new LinearInterpolator();
      case NodeClass::semi_lagrangian:
        return (Conf::Node*) new SemiLagrangian();
      case NodeClass::lddmm:
        return (Conf::Node*) new LDDMMTransformationModel();
      case NodeClass::singularity:
        return (Conf::Node*) new SingularityTransformationModel<float>();
      case NodeClass::registration:
        return (Conf::Node*) new Registration();
      case NodeClass::registration_frame:
        return (Conf::Node*) new RegistrationFrame();
      case NodeClass::point_singularity:
        return (Conf::Node*) new PointSingularity<float>();
      case NodeClass::line_singularity:
        return (Conf::Node*) new LineSingularity<float>();
      case NodeClass::curve_singularity:
        return (Conf::Node*) new CurveSingularity<float>();
      case NodeClass::quadrilateral_singularity:
        return (Conf::Node*) new QuadrilateralSingularity<float>();
      case NodeClass::curved_quadrilateral_singularity:
        return (Conf::Node*) new CurvedQuadrilateralSingularity<float>();
      default:
        throw invalid_argument("Unknown class keyword: " + str);
      }
    }catch(out_of_range e){
      throw invalid_argument("Unknown class keyword: " + str);
    }
  }

}
#endif
