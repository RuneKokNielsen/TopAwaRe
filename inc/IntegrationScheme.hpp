#ifndef INTEGRATION_SCHEME_HPP
#define INTEGRATION_SCHEME_HPP

#include "Image.hpp"
#include "Node.hpp"

/**
 * Abstract class for schemes for integrating velocity fields into transformations.
 **/
class IntegrationScheme : Conf::Node
{

public:

  virtual void init(ImageMeta meta) = 0;

  /**
   * Integrates the (possibly) time-dependent vector field f into
   * the transformation g.
   * \param f Time-dependent vector field to integrate
   * \param g Target transformation.
   * \param forward If true, computes the forward transformation T s.t. M = I \circ T. Otherwise computes T^{-1} s.t. Sinv = S \circ T^{-1}
   */
  virtual void integrate(const vector<vector<Image> > f, vector<vector<Image> > g, bool forward=true) = 0;

};
#endif
