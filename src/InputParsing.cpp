#include "InputParsing.hpp"

#include "OpenCVInt.hpp"
#include "NiftiInt.hpp"

#include "LinearInterpolator.hpp"
#include "CubicInterpolator.hpp"

#include "GaussianKernel.hpp"
#include "CauchyNavierKernel.hpp"

#include "SSD.hpp"
#include "L2alpha.hpp"

#include "LDDMMGradient.hpp"
#include "LDDMMTransformationModel.hpp"
#include "SemiLagrangian.hpp"

void InputParser::parse(){

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce this message")
    ("moving", po::value<string>()->required(), "path to moving image (required)")
    ("target", po::value<string>()->required(), "path to target image (required)")
    ("out", po::value<string>()->required(), "output directory (required)")
    ("normalize", po::value<bool>()->default_value(false), "normalize images (0/1)")
    ("padding", po::value<int>()->default_value(0), "image padding (>=0)")
    ("interpolator", po::value<int>()->default_value(0), "interpolator (0=linear, 1=cubic)")
    ("kernel", po::value<int>()->default_value(0), "kernel (0 = Gaussian, 1 = Cauchy-Navier )")
    ("kernel_smoothing", po::value<string>()->default_value("0.01"), "kernel smoothing (depends on kernel)")
    ("n", po::value<int>()->default_value(10), "timesteps (LDDMM)")
    ("t", po::value<double>()->default_value(1), "time (LDDMM)")
    ("lr", po::value<string>()->default_value("0.001"), "initial step learning rate")
    ("lr_min", po::value<string>()->default_value("0.0001"), "min learning rate")
    ("lr_max", po::value<string>()->default_value("0.1"), "max learning rate")
    ("lr_delta", po::value<double>()->default_value(10), "learning rate delta")
    ("lddmm_sigmas", po::value<string>()->default_value("0.01"), "lddmm sigma (regularization weight)")
    ("t_int", po::value<double>()->default_value(10), "timesteps (integration scheme)")
    ("display", po::value<int>()->default_value(0), "ms to display evolving image -- more time consuming")
    ("slice", po::value<string>(), "non-default bounding box. format: \"r0 r1 c0 c1 z0 z1\"")
    ("optimizer", po::value<int>()->default_value(0), "optimizer (0=lddmm, 1=lddmm+singularities)")
    ("singularities", po::value<string>(), "list of point singularities decimal coordinates (\"x1,y1,z1,x2,y2,z2,..\")")
    ("inv_singularities", po::value<string>(), "list of invert point singularities decimal coordinates (\"x1,y1,z1,x2,y2,z2,..\")")
    ("l_singularities", po::value<string>(), "list of line singularities decimal coordinates (\"x11,y11,z11,x12,y12,z12,x21,y21,z21,x22,y22,z22...\")")
    ("c_singularities", po::value<string>(), "list of curve singularities decimal coordinates (\"x11, y11, z11, x12, y12, z12,...")
    ("eta", po::value<double>()->default_value(0), "alpha channel penalization [0-1]")
    ("abs", po::value<double>()->default_value(1), "alpha boundary smoothing (>0)")
    ("maxits", po::value<int>()->default_value(0), "max iterations (0=no limit)")
    ("wendland_sigma", po::value<float>()->default_value(2), "Wendland RBF smoothing (singularity model")
    ("wendland_sigma_inv", po::value<float>()->default_value(2), "Wendland RBF smoothing for inversions (singularity model")
    ("singularity_movement", po::value<bool>()->default_value(false), "Singularities allowed to move")
    ("write_input", po::value<bool>()->default_value(false), "Write initialized input")
    ("scale_space", po::value<string>()->default_value("0"), "Scale space steps (\"sig1, sig2, sig3, ..\": <1 is full resolution)")
    ("cuda_device", po::value<int>()->default_value(0), "Cuda device id")
    ;

  po::store(po::parse_command_line(_argc, _argv, desc), _vm);

  if(_vm.count("help")){
    cout << desc << "\n";
    exit(0);
  }

  try{
    po::notify(_vm);
  }catch(po::required_option e){
    cout << "\nINPUT ERROR: " << e.what() << "\n\n" << desc << "\n";
    exit(1);
  }

  
}

template <> float InputParser::parseToken<float>(string token){
  return stof(token);
}
template <> int InputParser::parseToken<int>(string token){
  return stoi(token);
}

template<typename T>
vector<T> InputParser::parseVec(string str){
  vector<T> vals;
  size_t pos = 0;
  string token;
  while ((pos = str.find(",")) != string::npos) {
    token = str.substr(0, pos);
    vals.push_back(parseToken<T>(token));
    str.erase(0, pos + 1);
  }
  if(str.length() > 0){
    vals.push_back(stod(str));
  }
  return vals;
}

template<typename T>
Image InputParser::loadImage(string path){
  Image im;
  if(path.find(".nii")!=std::string::npos){
    im = NiftiInt::Nifti<T>::readImage(path);
  }else{
    im = OpenCVInt<T>::readImage(path);
  }

  if(_vm["normalize"].as<bool>()) ImageUtils::normalize(im);
  if(_vm.count("slice")){
    vector<int> slice = parseVec<int>(_vm["slice"].as<string>());
    if(slice.size() != 6){
      throw invalid_argument("slice input must contain exactly 6 points: r0, r1, c0, c1, z0, z1");
    }
    Cube cube = Cube(slice);
    im = ImageUtils::slice(im, cube);
  }

  return im;
}

Image InputParser::getMoving(){
  if(_movingLoaded) return _moving;
  _moving = loadImage<float>(getMovingPath());
  return _moving;
}
string InputParser::getMovingPath(){
  return _vm["moving"].as<string>();
}

Image InputParser::getTarget(){
  if(_targetLoaded) return _target;
  _target = loadImage<float>(getTargetPath());  
  return _target;
}
string InputParser::getTargetPath(){
  return _vm["target"].as<string>();
}

string InputParser::getOutPath(){
  return _vm["out"].as<string>();
}


bool InputParser::getWriteInput(){
  return _vm["write_input"].as<bool>();
}

int InputParser::getPadding(){
  return _vm["padding"].as<int>();
}

Interpolator* InputParser::getInterpolator(){
  switch(_vm["interpolator"].as<int>()){
  case linear:
    return new LinearInterpolator();
  case cubic:
    return new CubicInterpolator();
  default:
    throw invalid_argument("Invalid interpolator specified");
  }; 
}

Kernel* InputParser::getKernel(){
  vector<float> smoothing = parseVec<float>(_vm["kernel_smoothing"].as<string>());
  switch(_vm["kernel"].as<int>()){
  case gaussian:
    return new GaussianKernel<float>(smoothing);
  case cauchy_navier:
    return new CauchyNavierKernel<float>(smoothing);
  default:
    throw invalid_argument("Invalid smoothing kernel specified");
  }
}

double InputParser::getIntegrationT(){
  return _vm["t_int"].as<double>();
}


Metric* InputParser::getMetric(){
  switch(_vm["optimizer"].as<int>()){
  case lddmm:
    return new SSD<float>();
  case lddmm_singularities:
    return new L2alpha<float>(_vm["eta"].as<double>());
  default:
    throw invalid_argument("No metric found for specified optimizer");    
  }
}


void InputParser::loadSingularities(SingularityTransformationModel<float>* opt){

  ImageMeta meta = getMoving().meta();
  if(_vm.count("singularities")){

      
    vector<float> coords = parseVec<float>(_vm["singularities"].as<string>());
    int n = 3;
    if(coords.size() % n != 0){
      
      throw invalid_argument("Number of point singularities coordinates must be divisible by 3 but " + to_string(coords.size()) + " given\n");
    }
    for(int i=0; i<coords.size() / n; i++){
      Singularity<float> *s = new PointSingularity<float>(
	{coords.at(i*n),coords.at(i*n+1),coords.at(i*n+2),(float) 0.5/meta.height, _vm["wendland_sigma"].as<float>()});
      opt->addSingularity(s);	
    }
  }
  if(_vm.count("inv_singularities")){
    vector<float> coords = parseVec<float>(_vm["inv_singularities"].as<string>());
    int n = 3;
    if(coords.size() % n != 0){
      throw invalid_argument("Number of invert singularities coordinates must be divisible by 3 but " + to_string(coords.size()) + " given\n");
    }
    for(int i=0; i<coords.size() / n; i++){
      Singularity<float> *s = new PointSingularityInv<float>(
	{coords.at(i*n),coords.at(i*n+1),coords.at(i*n+2),(float) 0.5/meta.height, _vm["wendland_sigma_inv"].as<float>()});
      opt->addSingularity(s);
    }
  }
  if(_vm.count("l_singularities")){            
    vector<float> coords = parseVec<float>(_vm["l_singularities"].as<string>());
    int n = 6;
    if(coords.size() % n != 0){
      throw invalid_argument("Number of singularities coordinates must be divisible by 6 but " + to_string(coords.size()) + " given\n");
    }
    for(int i=0; i<coords.size() / n; i++){
      opt->addSingularity(new LineSingularity<float>({coords.at(i*n), coords.at(i*n+1), coords.at(i*n+2),
	      coords.at(i*n+3), coords.at(i*n+4), coords.at(i*n+5), (float) 0.5 / meta.height, _vm["wendland_sigma"].as<float>()}));
    }
  }

  if(_vm.count("c_singularities")){
    vector<float> coords = parseVec<float>(_vm["c_singularities"].as<string>());
    int n = 3;
    if(coords.size() % n != 0 || coords.size() < 6){
      throw invalid_argument("Number of singularities coordinates must be divisible by 3 and at least 6 but " + to_string(coords.size()) + " given\n");
    }
    // @TODO: For the moment we just assume a single curve..
    opt->addSingularity(new CurveSingularity<float>(coords, (float) 0.5/meta.height, _vm["wendland_sigma"].as<float>()));
  }
}

TransformationModel* InputParser::getOptimizer(){

  if(_optimizer != NULL) return _optimizer;
  
  GradientAlg* lddmmGrad = new LDDMMGradient<float>(parseVec<float>(_vm["lddmm_sigmas"].as<string>()), getKernel());
  switch(_vm["optimizer"].as<int>()){
  case lddmm:
    _optimizer = new LDDMMTransformationModel(lddmmGrad, getIntegrationScheme(), _vm["n"].as<int>(),
			      _vm["t"].as<double>(), parseVec<float>(_vm["lr"].as<string>()), getInterpolator());
    break;
  case lddmm_singularities:
    {
      TransformationModel* lddmmOpt = new LDDMMTransformationModel(lddmmGrad, getIntegrationScheme(), _vm["n"].as<int>(),
					       _vm["t"].as<double>(), parseVec<float>(_vm["lr"].as<string>()), getInterpolator());
      SingularityTransformationModel<float>* singuopt = new SingularityTransformationModel<float>(lddmmOpt, getInterpolator(), getMetric(), _vm["abs"].as<double>(), _vm["wendland_sigma"].as<float>(), _vm["singularity_movement"].as<bool>());

      loadSingularities(singuopt);

      _optimizer = singuopt;
      break;
    }
  default:
    throw invalid_argument("Invalid optimizer specified");
    
  }
  return _optimizer;
}

IntegrationScheme* InputParser::getIntegrationScheme(){
  return new SemiLagrangian(getInterpolator(), 1.0/getIntegrationT());
}


int InputParser::getDisplay(){
  return _vm["display"].as<int>();
}

int InputParser::getMaxIterations(){
  return _vm["maxits"].as<int>();
}

vector<float> InputParser::getLRMin(){
  return parseVec<float>(_vm["lr_min"].as<string>());
}

vector<float> InputParser::getLRMax(){
  return parseVec<float>(_vm["lr_max"].as<string>());
}
vector<float> InputParser::getScales(){
  return parseVec<float>(_vm["scale_space"].as<string>());
}

float InputParser::getLRDelta(){
  return _vm["lr_delta"].as<double>();
}

int InputParser::getCudaDevice(){
  return _vm["cuda_device"].as<int>();
}

int InputParser::getTransformationModelID(){
  return _vm["optimizer"].as<int>();
}
