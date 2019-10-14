# TopAwaRe
This is the reference GPU-implementation of the TopAwaRe (Topology-Aware Registration) framework published along [the paper](https://link.springer.com/chapter/10.1007%2F978-3-030-32245-8_41).

In addition to fast state-of-the-art diffeomorphic registration accelerated by GPU processing, this implementation, based on the TopAwaRe framework, allows handling a certain type of topological difference in the matching process.
In this piecewise-diffeomorphic extension, *holes* are allowed to grow from predefined discontinuities describing the topology of the source image.
The discontinuities range from a single point to curves and curved surfaces. For more information, please see [the paper](https://link.springer.com/chapter/10.1007%2F978-3-030-32245-8_41).

When relevant, please cite:\
*Nielsen R.K., Darkner S., Feragen A. (2019) TopAwaRe: Topology-Aware Registration. In: Shen D. et al. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2019. MICCAI 2019. Lecture Notes in Computer Science, vol 11765. Springer, Cham*





## Build and installation ##
### Supported systems ###
The appplication has been developed and tested on Ubuntu 16.04 systems, but should be compatible with more Linux distributions.
The codebase is primarily written in C++11 but relies on CUDA for GPU-processing.
In addition to the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) and a few libraries, the build requires sources from two other projects to facilitate [nifti](http://niftilib.sourceforge.net/) and [json](https://github.com/nlohmann/json) support. These sources are automatically downloaded during the build.

### Prerequisites ###
The build particularly requires the OpenCV library (*libopencv-dev*) and a couple of Boost libraries (program_options, assign and stacktrace (optional) - all contained in *libboost-all-dev*).

### Installation ###
Once all preqruisites are fulfilled, the installation is trivial. To build the application:
```
cd PATH_TO_REPOSITORY
cmake .
make
```
and optionally install it:
```
sudo make install
```

### Docker example ###
For reference, this repository includes a Dockerfile that builds the application inside a fresh image based on [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/).


## Running ##
After installing the software it can be used to run registration procedures from the command line.
A single registration procedure is run using the command:
```
topaware --source path_to_source_image --target path_to_target_image \
--out path_to_output_directory --configuration path_to_configuration_file
```
The first three arguments are self-explanatory, but the *--configuration* argument less so. This software is aiming at high flexibility, which in turn results in additional complexity. What the last argument refers to is a file in JSON-format describing a computation tree consisting of multiple configurable nodes all describing some part of the registration procedure.


### Configuration Tree ###
We start by describing the various nodes and their relations, refering to the diagram below.
#### Diagram ####
This diagram illustrates the components of the computational tree including their relations and configurable attributes. Each node corresponds to an actual C++ class and the relations between the resulting objects are one-to-one translatable to the configuration data. The attributes within the nodes are only those configurable by the user. The relations in the diagram can be split into:
- Composition (black diamond) signifies a parent-child relationship (e.g. a **Registration** contains a **Measure** object).
- Implementation (dashed arrow) signifies the implementation of an interface (e.g. **CC** is a type of **Measure**).
To properly configure a **Registration** node, the user must thus specify a **Measure** child node, which could be of type **CC**.

The purpose of each node and its attributes are given below.

![configuration_diagram](https://github.com/RuneKokNielsen/TopAwaRe/blob/master/media/topaware.png)
#### Nodes & Attributes ####
##### RegistrationFrame ####
The **RegistrationFrame** is the root node in the execution tree. Its responsibility is to initialize the registration and store the results. 
###### Attributes ######
- normalize : bool *[Optional]* -> Whether initial images should be normalized to the [0, 1] interval. Defaults to false.
- slice : int[] *[Optional]* -> Allows registering a specific slice of the images. The format is [r0, r1, c0, c1, z0, z1] corresponding to start (inclusive) and end (exclusive) pixel indeces in row-major order. Defaults to entire images.
- padding : int *[Optional]* -> Allows padding an image with zeros in all dimensions. Defaults to 0.
- cuda_device : int *[Optional]* -> Specify the cuda device id. Defaults to 0.
- registration : **Registration** -> Specify a subtree which must be of type **Registration**.

##### Registration ####
Controls a registration procedure.

###### Attributes ######
- scale : int[] -> Defines the scales to be registered. Each scale is defined by an integer corresponding to the resolution reduction (in all dimensions) as power of two. For instance, 0 is initial scale (e.g. 100x100x100), 1 is half scale (50x50x50) and 2 is quarter scale (25x25x25). **NB**: The levels are popped from the right, so [0, 1, 2] would correspond to registration going from quarter scale -> half scale -> full scale. The same is true for other similar attributes.
- maxits : int[] -> Defines the maximum number of iterations at each scale. 0 for unlimited iterations.
- min_convergence_rate *[Optional]* : float -> Sets the minimum convergence rate. The registration at current scale is ended after iteration *i* if the measure *e_i* is greater than *min_covergence_rate * e_{i-10}*. Defaults to 0.999.
- transformation_model : **TransformationModel** : Specify a subtree of type **TransformationModel** (e.g. **LDDMMTransformationModel** or **SingularityTransformationModel**. Defines the underlying registration algorithm.
- measure : **Measure** : Specify a subtree of type **Measure** (e.g. **CC**). Defines the objective of the optimization.
- interpolator : **Interpolator** : Specify a subtree of type **Interpolator** (e.g. **LinearInterpolator**). Used for resampling images after each iteration and when changing scale.

##### CC (Measure) #####
Zero-normalized cross correlation. This is currently the only standard measure supplied with the software. 
###### Attributes ######
- w : int *[Optional]* -> Controls the width of the convolutions measured in pixels from the center. For w=2 the convolution window in 3D would thus be 5x5x5=125. Defaults to 2.

##### CCalpha (Measure) #####
Extension of **CC** that allows accounting for background intensity. This is required for the **SingularityTransformationModel**.
##### Attributes #####
- w : int *[Optional]* -> See **CC**:w.
- beta : int *[Optional]* -> Controls the background intensity. Defaults to 0.

##### LDDMMTransformationModel : TransformationModel #####
Implementation of the LDDMM which models the transformation over smooth, time-dependent velocity fields.
###### Attributes ######
- N : int *[Optional]* -> Number of timesteps in the discretization. Defaults to 10.
- elastic_sigma : float *[Optional]* -> Controls optional smoothing of the velocity field after each update emulating elastic effects. 0 for no smoothing. Defaults to 0.
- reparameterize : int *[Optional]* -> Number of iterations between constant-speed reparameterization of velocity fields. 0 to disable. Defaults to 0.
- grad_scale : float *[Optional]* -> Size in voxel width of maximal velocity field update magnitude. After computing the gradient w.r.t. the velocity field, the gradient is rescaled so that its greatest update corresponds to this value. Defaults to 0.2.
- kernel : **Kernel** : Specify a subtree of type **Kernel** used to smoothen the gradients.
- sigma : float[] : Controls the balance (at each scale) between regularization and similarity.
- integration_scheme : **IntergrationScheme** : Specify a subtree of type **IntegrationScheme** used to integrate the time-dependent velocity fields into warps.
- interpolator : **Interpolator** : Specify a subtree of type **Interpolator** used to resample velocity fields when changing scale.

##### CauchyNavierKernel : Kernel #####
A Cauchy-Navier smoothing kernel based on elastic equations *gamma I + alpha x nabla^2*.
###### Attributes ######
- alpha_gamma : float[] -> Smoothing factor at each scale such that *gamma = 1/(1+alpha_gamma)* and *alpha = 1 - gamma*. Larger values results in more smoothing.

##### GaussianKernel : Kernel #####
Gaussian smoothing kernel.
###### Attributes ######
- sigma : float[] -> Smoothing factor at each scale.

##### SemiLagrangian : IntegrationScheme #####
Semi-Lagrangian integration scheme for velocity field integration.
###### Attributes ######
- delta_t : float *[Optional]* -> Time step size. Defaults to 0.1.
- interpolator : **Interpolator** -> Specify a subtree of type **Interpolator**. Used for resampling during integration.

##### SingularityTransformationModel : TransformationModel #####
Transformation model for modeling singularities. Since it only models any supplied singularities, it must also be supplied with a base transformation model to optimize the global transformation.
###### Attributes ######
- h : float *[Optional]* -> Controls size of step when estimating gradients of expansion radii. Defaults to 0.1.
- a : float *[Optional]* -> Controls the gradient scaling. Defaults to 0.01.
- alpha_boundary_smoothing : float *[Optional]* -> Controls the smoothness of the transition from foreground to background close to the true boundaries between expansions and regular domain. Given in voxel width. Defaults to 1.
- opt_proportion : int *[Optional]* -> Often the global optimization requires more steps than the expansions. This values controls the proportion of expansion steps versus base model steps. One step with the singularity model will be taken for every *opt_proportion* steps with the base model. Defaults to 1.
- opt_proportion_increase : int *[Optional]* -> In some cases the expansions can be optimized during early optimization at coarse scales. This value allows increasing *opt_proportion* at every *opt_proportion_increase* iteration. 0 to disable. Defaults to 0.
- curves_dir : string *[Situational]* -> When optimizing curved singularities, their psi-diffeomorphisms can optionally be cached for later use. If this is the case, they will be stored (and loaded from) the directory specified here.
- interpolator : **Interpolator** -> Specify a subtree of type **Interpolator**. Used for computing *W_alpha* deformation.
- transformation_model : **TransformationModel** -> Specify a subtree of type **TransformationModel**. Used to optimize the the global diffeomorphism.
- psi_transformation_model :: **TransformationModel** *[Situational]* -> Specify a subtree of type **TransformationModel**. When optimizing curved singularities, this model is used to find their *psi*-diffeomorphisms.
- singularities : **Singularity[]** -> Specify a list of **Singularity** subtrees. These are the singularities to be optimized.

##### Singularity #####
All singularities share a couple of basic attributes.
- pi : float *[Optional]* -> Starting expansion radius measured in image size. Defaults to 0.0001.
- sigma : float *[Optional]* -> Displacement smoothing factor. Defaults to 2.
- max_pi : float *[Optional]* -> Max expansion radius. Defaults to 1.

##### PointSingularity : Singularity #####
The simplest singularity type. A sphere growing from a single point.
###### Attributes ######
- point : float[] -> Coordinates [r,c,z] given in [0,1] intervals defining the center of the singularity.

##### LineSingularity : Singularity #####
Expansion from a straight line.
##### Attributes ######
- points : float[][] -> Array of two coordiantes [[r0,c0,z0], [r1,c1,z1]] describing end points.

##### QuadrilateralSingularity : Singularity #####
Expansion from a quadrilateral.
##### Attributes #####
- points : float[][] -> Array of three coordinates [p0=[r0,c0,z0], p1=[r1,c1,z1], p2=[r2,c2,z2]] defining the quadrilateral as the parallelogram contained in p0, p1, p0+(p1-p0)+(p2-p0), p0.

##### CurveSingularity : Singularity #####
Expansion from a curve.
###### Attributes ######
- points : float[][] -> Array of n >= 2 coordinates on the curve.
- maxits : int -> Number of iterations when optimizing the diffeomorphism psi that straightens the curve.
- cache_key : string *[Optional]* -> If a key is specified; attempt to load a cached version of psi or store the result of the psi optimization.

##### CurvedQuadrilateralSingularity : Singularity #####
Expansion from a curved quadrilateral.
##### Attributes #####
- points : float[][][] -> 2D array of coordinates describing the surface. The number of columns must be equal across all rows.
- maxits : int -> See CurveSingularity:maxits
- cache_key : string -> See CurveSingularity:cache_key

#### Examples #####
As a starting point, we here supply configuration files for various registration procedures.

##### Diffeomorpic #####
This is an example of multiscale diffeomorphic reigstration. It applies LDDMM with 4 timesteps at 3 scales, up to 100 iterations at each layer.
```
{
  "root":{
	  "class": "registration_frame",
	  "normalize": true,	 
	  "registration": {
	    "class": "registration",	  
	    "transformation_model":{
		    "class": "lddmm",
		    "sigma": [ 0,0,0],
		    "kernel":{
		      "class": "cauchy_navier",
		      "alpha_gamma": [0.005, 0.005, 0.005]
		    },
		    "integration_scheme": {
		      "class": "semi_lagrangian",
		      "delta_t": 0.25,
		      "interpolator":{
			      "class": "linear"
		      }
		    },
		    "N": 4,		   
		    "interpolator":{
		      "class": "linear"
		    }
	    },
	    "interpolator":{
		    "class": "linear"
	    },
	    "measure":{
		    "class": "cc",
		    "w": 4
	    },	   
	    "maxits": [100, 100, 100],
	    "scale": [0, 1, 2]
	  }
  }
}

```
##### Piecewise-Diffeomorphic #####
Below is a full example where the singularity model is used to optimize expansion of two curved quadrilateral singularities in addition to the normal LDDMM registration. The surface control points are stored in external files and included using a special "include" keyword. Examples of the singularities themselves are provided below.
```
{

  "root":{
	  "class": "registration_frame",	
	  "normalize": true,
	  "registration": {
	    "class": "registration",	    
	    "transformation_model":{
		    "class": "singularity",
		    "a": 10,
		    "h": 0.1,
		    "opt_proportion": 1,
        "opt_proportion_increase": 10,
		    "singularities":[
		      {
			      "class": "curved_quadrilateral_singularity",
			      "include": "singularity_rh.json",
			      "sigma": 2,
			      "maxits": 100,
			      "max_pi": 0.02,
			      "pi": 0.01,
			      "cache_key": "curve_rh"
		      },
		      {
			      "class": "curved_quadrilateral_singularity",
			      "include": "singularity_lh.json",
			      "sigma": 2,
			      "maxits": 100,
			      "max_pi": 0.02,
			      "pi": 0.01,
			      "cache_key": "curve_lh"
		      }	
		    ],
        "curves_dir": "curves",
		    "psi_transformation_model": {
		      "class": "lddmm",
		      "elastic_sigma": 0.25,
		      "sigma": [0.000],
		      "kernel":{
			      "class": "cauchy_navier",
			      "alpha_gamma": [0.0001]
		      },
		      "integration_scheme":{
			      "class": "semi_lagrangian",
			      "delta_t": 0.25,
			      "interpolator":{
			        "class": "linear"
			      }
		      },
		      "N": 4,
		      "interpolator": {
			      "class": "linear"
		      }
		    },
		    "interpolator":{
		      "class": "linear"
		    },
		    "transformation_model":{
		      "class": "lddmm",
		      "sigma": [0,0,0],
		      "kernel":{
			      "class": "cauchy_navier",
			      "alpha_gamma": [0.005, 0.005, 0.005]
		      },
		      "integration_scheme": {
			      "class": "semi_lagrangian",
			      "delta_t": 0.25,
			      "interpolator":{
			        "class": "linear"
			      }
		      },
		      "N": 4,		     
		      "interpolator":{
			      "class": "linear"
		      }
		    },
		    "alpha_boundary_smoothing": 0.5
	    },
	    "interpolator":{
		    "class": "linear"
	    },
	    "measure":{
		    "class": "cc_alpha",
		    "beta": 0.2,
		    "w": 4
	    },
	    "maxits": [100,100,100],
	    "scale": [0, 1, 2]
	  }
  }
}
```
##### Point #####
```
"singularities":[
  {
    "class": "point_singularity",
    "point": [0.5, 0.5, 0.5]
  }
]
```
##### Line segment #####
```
"singularities":[
  {
    "class": "line_singularity",
    "points": [[0.2 , 0.3, 0.4], [0.3, 0.4, 0.5]]
  }
]
```
##### Quadrilateral #####
```
"singularities":[
  {
    "class": "quadrilateral_singularity",
    "points": [[0, 0.5, 0], [1, 0.5, 0], [0, 0.5, 1]]
  }
]
```
##### Curve #####
```
"singularities":[
  {
    "class": "curve_singularity",
    "points": [
        [0.48,0.3,0.3],
        [0.48,0.3,0.38],
        [0.48, 0.31, 0.44],
        [0.48,0.37,0.47],
        [0.48, 0.45, 0.48],
        [0.48, 0.49, 0.44],
        [0.48, 0.56, 0.41],
        [0.48, 0.63, 0.44],
        [0.48, 0.66, 0.48],
        [0.48, 0.67, 0.57],
        [0.48, 0.65, 0.65],
        [0.48, 0.67, 0.70]
     ]
  }
]
```
##### Curved Quadrilateral #####
```
"singularities":[
  {
    "class": "curved_quadrilateral_singularity",
    "points": [
        [[0,0.5,0], [0, 0.75, 0.75], [0,0.5,1]],
        [[0.1,0.51,0], [0.1, 0.76, 0.75], [0.1,0.51,1]],
        [[0.2,0.52,0], [0.2, 0.77, 0.75], [0.2,0.52,1]],
        [[0.3,0.53,0], [0.3, 0.78, 0.75], [0.3,0.53,1]],
        [[0.4,0.54,0], [0.4, 0.79, 0.75], [0.4,0.54,1]],
        [[0.5,0.55,0], [0.5, 0.80, 0.75], [0.5,0.55,1]],
        [[0.6,0.56,0], [0.6, 0.81, 0.75], [0.6,0.56,1]],
        [[0.7,0.57,0], [0.7, 0.82, 0.75], [0.7,0.57,1]],
        [[0.8,0.58,0], [0.8, 0.83, 0.75], [0.8,0.58,1]],
        [[0.9,0.59,0], [0.9, 0.84, 0.75], [0.9,0.59,1]],
        [[1,0.6,0], [1, 0.85, 0.75], [1,0.6,1]]
      ]
  }
]
```
