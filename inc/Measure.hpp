#ifndef MEASURE_HPP
#define MEASURE_HPP

#include "Node.hpp"
#include <Image.hpp>
#include "Interpolator.hpp"


namespace measure{

  /**
   * A Context holds time-local variables necessary for locally
   * computing a measure and its gradients.
   **/
  struct Context{
    /**
     * \param timesteps The number of timesteps within this context.
     **/
    Context(uint timesteps): _timesteps(timesteps){
    };

    /**
     * Get the context at given timestep.
     * \param timestep The timestep of the target context.
     **/
    virtual Context* getContext(uint timestep) = 0;

    // Number of timesteps of the context.
    uint _timesteps;
    // Interpolator object which might be required for computations.
    Interpolator* _interpolator;

    virtual ~Context(){
    }

    /**
     * Initialize the context with required objects.
     * \param S The target image.
     * \param I The source image.
     * \param interpolator Interpolator object to be used within this context.
     **/
    virtual void init(Image S, Image I, Interpolator* interpolator){
      _I = I;
      _S = S;
      _interpolator = interpolator;
    }

    /**
     * Update the context based on the current transformation at the
     * timestep of this context.
     **/
    virtual void update(vector<Image> phi, vector<Image> phiInv){
      // Surpress unused-warning of optional virtual function
      (void) phi; (void) phiInv;
    };

    // Reference to the source image.
    Image _I;
    // Reference to the target image.
    Image _S;
    /**
     * Holds the individual contexts for all timesteps, this object
     * corresponding to the first element.
     **/
    vector<Context*> _subs;
  };

  /**
   * A Measure describes how to compute a given similarity measure
   * as well as its gradients within a given Measure::Context.
   **/
  class Measure : Conf::Node
  {

  public:
    /**
     * Given a context, computes the similarity measure value.
     * \param context The context to compute the measure in.
     **/
    virtual double measure(const Context* context) const = 0;

    /**
     * Given a context, computes the gradient of the similarity measure.
     * \param context The context to compute the gradient in.
     **/
    virtual vector<Image> gradient(const Context* context) const = 0;

    /**
     * Returns the default context for a dsicretization with
     * given number of timesteps.
     * \param timesteps The number of timesteps.
     **/
    virtual Context *baseContext(uint timesteps) const = 0;

  };

  /**
   * An AlphaContext is any Measure::Context that somehow handles
   * an alpha channel. This is required for applying the
   * topology-aware parts of the implementation.
   **/
  struct AlphaContext : virtual Context{

    AlphaContext(): Context(1){
    }

    virtual ~AlphaContext(){
    }

    /**
     * Returns a copy of itself that can be modified without changing
     * anything within this context.
     **/
    virtual AlphaContext* copy() = 0;

    virtual AlphaContext *getContext(uint t){ return t==0?this:dynamic_cast<AlphaContext*>(_subs.at(t)); }


    /**
     * One-sided update.
     * This function is used for finite difference approximation
     * of the gradient w.r.t. changes in the singularity model.
     * Updates the context using a full transformation.
     * Note that the alpha field within this context has already
     * accounted for this transformation and as such should not
     * be transformed again!
     * \param phi Transformation to apply.
     **/
    virtual void updateAlpha(vector<Image> phi) = 0;

    /**
     * Current alpha field.
     **/
    Image alpha;

  };

}

#endif
