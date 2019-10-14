#include "Singularities.hpp"
#include "math.h"
#include <limits>

/**
 * ======================================================
 * POINT SINGULARITY
 * ======================================================
 **/

template<typename T>
void PointSingularity<T>::fillAlpha(T* palpha, T** X, ImageMeta meta, float abs) const{
  double pi = this->getPi();
  
#ifdef CUDA_GPU
  if(meta.m == gpu){
    cuda_pointSingularity_fillAlpha(meta.ncells, meta.dimensions, (float**) X, (float*) palpha, (float*) this->X, pi, abs);
    return;
  }
#endif
  NOT_IMPLEMENTED
}

template<typename T>
void PointSingularity<T>::fillWalpha(T** XW, T** X, ImageMeta meta) const{

#ifdef CUDA_GPU
  if(meta.m == gpu){
    cuda_pointSingularity_fillWalpha(meta.ncells, meta.dimensions, (float**) XW, (float**) X, (float*) this->X, this->getPi(), this->getSigma());
    return;
  }
#endif
  NOT_IMPLEMENTED;
}

template<typename T>
void PointSingularity<T>::fillWalphaInv(T** XW, T** X, ImageMeta meta) const{
#ifdef CUDA_GPU
  cuda_pointSingularity_fillWalphaInv(meta.ncells, meta.dimensions, (float**) XW, (float**) X, (float*) this->X, this->getPi(), this->getSigma());
  return;
#endif
  NOT_IMPLEMENTED;
}


/**
 * ======================================================
 * LINE SINGULARITY
 * ======================================================
 **/

template<typename T>
void LineSingularity<T>::fillAlpha(T* palpha, T** X, ImageMeta meta, float abs) const{


#ifdef CUDA_GPU
  if(meta.m == gpu){
    cuda_lineSingularity_fillAlpha(meta.ncells, meta.dimensions, (float**) X, (float*) palpha, (float*) this->X, this->getPi(), abs, lambda);
    return;
  }
#endif
  NOT_IMPLEMENTED;
}

template<typename T>
void LineSingularity<T>::fillWalpha(T** XW, T** X, ImageMeta meta) const{


#ifdef CUDA_GPU
  if(meta.m == gpu){
    cuda_lineSingularity_fillWalpha(meta.ncells, meta.dimensions, (float**) XW, (float**) X, (float*) this->X, this->getPi(), this->getSigma(), lambda);
    return;
  }
#endif
  NOT_IMPLEMENTED;
}

template<typename T>
void LineSingularity<T>::fillWalphaInv(T** XW, T** X, ImageMeta meta) const{
#ifdef CUDA_GPU
  cuda_lineSingularity_fillWalphaInv(meta.ncells, meta.dimensions, (float**) XW, (float**) X, (float*) this->X, this->getPi(), this->getSigma(), lambda);
  return;
#endif

  NOT_IMPLEMENTED;
}

/**
 * ======================================================
 * CURVE SINGULARITY
 * ======================================================
 **/

float length_and_center_of_gravity(vector<float> coords, float c[3]){
  float len = 0;
  for(uint i=0; i<coords.size()/3-1; i++){
    float squared_norm_i = 0;
    float cx_i[3];
    for(uint j=0; j<3; j++){
      cx_i[j] = (coords.at(i*3+j) + coords.at((i+1)*3+j)) / 2;
      float dj = coords.at((i+1)*3+j) - coords.at(i*3+j);
      squared_norm_i += dj * dj;
    }
    float len_i = sqrt(squared_norm_i);
    for(uint j=0; j<3; j++){
      c[j] += cx_i[j] * len_i;
    }
    len += len_i;
  }
  for(uint j=0; j<3; j++){
    c[j] = c[j] / len;
  }
  return len;
}

vector<float> supersample(vector<float> coords, int N){

  float cx[3];
  float len = length_and_center_of_gravity(coords, cx);

  vector<float> res;
  float x[3];
  float x0[3];
  float x1[3];
  float x0x1[3];
  float x0x1norm = 0;
  int iNext = 1;

  float seglen = len / (float) N;
  for(int j=0; j<3; j++){
    res.push_back(coords.at(j));
    x[j] = coords.at(j);
    x0[j] = x[j];
    x1[j] = coords.at(3+j);
    x0x1[j] = x1[j]-x0[j];
    x0x1norm += x0x1[j]*x0x1[j];
  }

  x0x1norm = sqrt(x0x1norm);

  for(int i=1; i<N; i++){
    float l = seglen;
    float dist = 0;
    for(int j=0; j<3; j++){
      dist += (x1[j]-x[j])*(x1[j]-x[j]);
    }
    dist = sqrt(dist);
    while(dist < l){

      l = l - dist;
      for(int j=0; j<3; j++){
	x[j] = x1[j];
      }
      if(i == N-1){
	for(int j=0; j<3; j++){
	  res.push_back(x[j]);
	}
	return res;
      }
      
      x0x1norm = 0;
      for(int j=0; j<3; j++){
	x0[j] = coords.at(3*iNext + j);
	x1[j] = coords.at(3*(iNext+1) + j);
	x0x1[j] = x1[j]-x0[j];
	x0x1norm += x0x1[j]*x0x1[j];
      }
      x0x1norm = sqrt(x0x1norm);
      iNext++;

      dist = 0;
      for(int j=0; j<3; j++){
	dist += (x1[j]-x[j])*(x1[j]-x[j]);
      }
      dist = sqrt(dist);
    }

    for(int j=0; j<3; j++){
      x[j] = x[j] + l * x0x1[j]/x0x1norm;
      res.push_back(x[j]);
    }
  }
  return res;
}

template<typename T>
void CurveSingularity<T>::determineLine(){
  
  int N = _coords.size() / 3;

  // Find points corresponding to edge length preserverving deformation

  // Compute center of gravity
  float cx[3] = {0};
  
  // Compute mean direction
  float dx[3] = {0};
  float normdx2 = 0;

  for(int j=0; j<3; j++){
    dx[j] = _coords.at((N-1)*3+j) - _coords.at(j);
    normdx2 += dx[j] * dx[j];
  }

  for(int j=0; j<3; j++){
    dx[j] = dx[j] / sqrt(normdx2);
    _lineCoords.push_back(0);
  }

  // Compute straight curve control points preserving edge lengths
  for(int i=1; i<N; i++){
    float normi2 = 0;
    for(int j=0; j<3; j++){
      float dj = _coords.at(i*3+j) - _coords.at((i-1)*3+j);
      normi2 += dj*dj;
    }
    for(int j=0; j<3; j++){
      _lineCoords.push_back(_lineCoords.at((i-1)*3+j) + sqrt(normi2) * dx[j]);
    }
  }

  // Align center of gravity
  float cy[3] = {0};
  length_and_center_of_gravity(_lineCoords, cy);
  for(int i=0; i<N; i++){
    for(int j=0; j<3; j++){
      _lineCoords.at(i*3+j) = _lineCoords.at(i*3+j) + (cx[j] - cy[j]);
      //cout << _lineCoords.at(i*3+j) << ", ";
    }
  }

  // Sample evenly spaced points
  _coords = supersample(_coords, 100);
  _lineCoords = supersample(_lineCoords, 100);  

}


template<typename T>
void CurveSingularity<T>::projectToLine(float *p0, float *p1, float *x, float *xl){

  float p0_x[3];
  float p0_p1[3];
  float norm2 = 0;
  
  for(int i=0; i<3; i++){
    p0_x[i] = x[i] - p0[i];
    p0_p1[i] = p1[i] - p0[i];
    norm2 += p0_p1[i] * p0_p1[i];
  }

  float norm = sqrt(norm2);
  float t = 0;
  for(int i=0; i<3; i++){
    t += p0_x[i] * p0_p1[i] / norm;
  }

  for(int i=0; i<3; i++){
    xl[i] = p0[i] + t * p0_p1[i] / norm;
  }    

}

template<typename T>
void CurveSingularity<T>::projectToLine(float* w, float* x, float* xl){

  float p0[3] = {0};
  float p1[3] = {0};

  p0[0] = w[2];

  p1[0] = w[0] + w[1] + w[2];
  p1[1] = 1;
  p1[2] = w[1] == 0 ? 0 : 1;


  projectToLine(p0, p1, x, xl);

}


template<typename T>
vector<Image> CurveSingularity<T>::curvedToMesh(){
  return curveToMesh(_coords);
}

template<typename T>
vector<Image> CurveSingularity<T>::straightToMesh(){
  return curveToMesh(_lineCoords);
}

template<typename T>
vector<Image> CurveSingularity<T>::curveToMesh(vector<float> curve){

  vector<Image> mesh;
  ImageMeta meta(curve.size() / 3, 1, 1, 2, f32, 4);
  for(uint i=0; i<3; i++){
    mesh.push_back(Image(meta));
    T* ptr = mesh.at(i).ptr<T>();
    for(uint j=0; j<curve.size() / 3; j++){
      ptr[j] = curve.at(j*3 + i);
    }
#ifdef CUDA_GPU
    mesh.at(i).toGPU();
#endif
  }

  return mesh;
  
}


template<typename T>
void CurveSingularity<T>::updateCurvedCoords(Image mesh, int dim){
#ifdef CUDA_GPU
  mesh.toCPU();
#endif
  float* ptr = mesh.ptr<float>();
  for(uint k=0; k<nCoords(); k++){
    _coords.at(k*3 + dim) = ptr[k];
  }
#ifdef CUDA_GPU
  mesh.toGPU();
#endif
}

template<typename T>
void CurveSingularity<T>::updateStraightCoords(Image mesh, int dim){
#ifdef CUDA_GPU
  mesh.toCPU();
#endif
  float* ptr = mesh.ptr<float>();
  for(uint k=0; k<nCoords(); k++){
    _lineCoords.at(k*3 + dim) = ptr[k];
  }
#ifdef CUDA_GPU
  mesh.toGPU();
#endif
}

template<typename T>
float CurveSingularity<T>::getCurvedCoord(int i, int dim){
  return _coords.at(i * 3 + dim);
}

template<typename T>
float CurveSingularity<T>::getStraightCoord(int i, int dim){
  return _lineCoords.at(i * 3 + dim);
}

/**
 * ======================================================
 * QUADRILATERAL SINGULARITY
 * ======================================================
 **/


template<typename T>
void QuadrilateralSingularity<T>::fillAlpha(T* palpha, T** X, ImageMeta meta, float abs) const{

  double pi = this->getPi();

#ifdef CUDA_GPU
  if(meta.m == gpu){
    cuda_planeSingularity_fillAlpha(meta.ncells, meta.dimensions, (float**) X, (float*) palpha, (float*) this->X, pi, abs, Binv, B, lambda);
    return;
  }
#endif

  throw logic_error("Not implemented: Quadrilateralsingularity fillalpha cpu");

}

template<typename T>
void QuadrilateralSingularity<T>::fillWalpha(T** XW, T** X, ImageMeta meta) const{

#ifdef CUDA_GPU
  if(meta.m == gpu){
    cuda_planeSingularity_fillWalpha(meta.ncells, meta.dimensions, (float**) XW, (float**) X, (float*) this->X, this->getPi(), this->getSigma(), Binv, B, lambda);
    return;
  }
#endif

  throw logic_error("Not implemented: Quadrilateralsingularity fillwalpha cpu");

}

template<typename T>
void QuadrilateralSingularity<T>::fillWalphaInv(T** XW, T** X, ImageMeta meta) const{
#ifdef CUDA_GPU
  cuda_planeSingularity_fillWalphaInv(meta.ncells, meta.dimensions, (float**) XW, (float**) X, (float*) this->X, this->getPi(), this->getSigma(), Binv, B, lambda);
  return;
#endif
  throw logic_error("Not implemented: Quadrilateralsingularity fillwalphainv cpu");

}

template<typename T>
void QuadrilateralSingularity<T>::init(){
  determineBasis();
}

template<typename T>
void QuadrilateralSingularity<T>::determineBasis(){

  // Fill basis
  for(int i=0; i<3; i++){
    for(int j=0; j<2; j++){
      B[i][j] = X[i + 3*(j+1)] - X[i];
    }
  }
  // Last column is unit normal vector
  float n[3];
  n[0] = B[1][0]*B[2][1] - B[2][0]*B[1][1];
  n[1] = B[2][0]*B[0][1] - B[0][0]*B[2][1];
  n[2] = B[0][0]*B[1][1] - B[1][0]*B[0][1];
  float norm2 = 0;
  for(int i=0; i<3; i++){
    norm2 += n[i]*n[i];
  }
  for(int i=0; i<3; i++){
    B[i][2] = n[i] / sqrt(norm2);
  }

  // Inversion..
  Binv[0][0] = B[1][1]*B[2][2] - B[1][2]*B[2][1];
  Binv[1][0] = -(B[1][0]*B[2][2] - B[1][2]*B[2][0]);
  Binv[2][0] = B[1][0]*B[2][1] - B[1][1]*B[2][0];
  Binv[0][1] = -(B[0][1]*B[2][2] - B[0][2]*B[2][1]);
  Binv[1][1] = B[0][0]*B[2][2] - B[0][2]*B[2][0];
  Binv[2][1] = -(B[0][0]*B[2][1] - B[0][1]*B[2][0]);
  Binv[0][2] = B[0][1]*B[1][2] - B[0][2]*B[1][1];
  Binv[1][2] = -(B[0][0]*B[1][2] - B[0][2]*B[1][0]);
  Binv[2][2] = B[0][0]*B[1][1] - B[0][1]*B[1][0];

  float det = B[0][0]*Binv[0][0] + B[0][1]*Binv[1][0] + B[0][2]*Binv[2][0];

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      Binv[i][j] /= det;
    }
  }

}

/**
 * ======================================================
 * CURVED QUADRILATERAL SINGULARITY
 * ======================================================
 **/

template<typename T>
void CurvedQuadrilateralSingularity<T>::determineQuadrilateral(){

  
  for(uint i=0; i<rows(); i++){
    for(uint j=0; j<cols(); j++){
      for(uint k=0; k<3; k++){
        _straightCoords.at(i).at(j).at(k) = this->X[k];
        _straightCoords.at(i).at(j).at(k) += ((float) i/(rows()-1))*this->B[k][1];
        _straightCoords.at(i).at(j).at(k) += ((float) j/(cols()-1))*this->B[k][0];
      }
      
    }
    
  }
}

template<typename T>
vector<Image> CurvedQuadrilateralSingularity<T>::curvedToMesh(){
  return curveToMesh(_curvedCoords);
}

template<typename T>
vector<Image> CurvedQuadrilateralSingularity<T>::straightToMesh(){
  return curveToMesh(_straightCoords);
}

template<typename T>
void CurvedQuadrilateralSingularity<T>::updateCurvedCoords(Image mesh, int dim){
#ifdef CUDA_GPU
  mesh.toCPU();
#endif
  float* ptr = mesh.ptr<float>();
  for(uint i=0; i<rows(); i++){
    for(uint j=0; j<cols(); j++){
      _curvedCoords.at(i).at(j).at(dim) = ptr[i*cols()+j];
    }
  }
#ifdef CUDA_GPU
  mesh.toGPU();
#endif
}

template<typename T>
void CurvedQuadrilateralSingularity<T>::updateStraightCoords(Image mesh, int dim){
#ifdef CUDA_GPU
  mesh.toCPU();
#endif
  float* ptr = mesh.ptr<float>();
  for(uint i=0; i<rows(); i++){
    for(uint j=0; j<cols(); j++){
      _straightCoords.at(i).at(j).at(dim) = ptr[i*cols()+j];
    }
  }
#ifdef CUDA_GPU
  mesh.toGPU();
#endif
}

template<typename T>
float CurvedQuadrilateralSingularity<T>::getCurvedCoord(int i, int dim){
  return _curvedCoords.at(i / cols()).at(i % cols()).at(dim);
}

template<typename T>
float CurvedQuadrilateralSingularity<T>::getStraightCoord(int i, int dim){
  return _straightCoords.at(i / cols()).at(i % cols()).at(dim);
}

template<typename T>
vector<Image> CurvedQuadrilateralSingularity<T>::curveToMesh(vector<vector<vector<float>>> curve){
  vector<Image> mesh;
  ImageMeta meta(rows(), cols(), 1, 2, f32, 4);
  for(uint i=0; i<3; i++){
    mesh.push_back(Image(meta));
    float* ptr = mesh.at(i).ptr<float>();
    for(uint j=0; j<rows(); j++){
      for(uint k=0; k<cols(); k++){
        ptr[j*cols()+k] = curve.at(j).at(k).at(i);
      }
    }
#ifdef CUDA_GPU
    mesh.at(i).toGPU();
#endif
  }
  return mesh;
}

template<typename T>
void CurvedQuadrilateralSingularity<T>::init(){
  // Use quadrilateral spanned by the three adjacent corners
  // and translated so the center of gravity is aligned with
  // the curved surface

  // Compute CoG for surface
  float cog_curve[3] = {0};
  for(uint i=0; i<rows(); i++){
    for(uint j=0; j<cols(); j++){
      for(uint k=0; k<3; k++){
        cog_curve[k] += _curvedCoords.at(i).at(j).at(k);
      }
    }
  }
  for(uint k=0; k<3; k++){
    cog_curve[k] /= rows() * cols();
  }

  float cog_straight[3] = {0};
  for(uint i=0; i<3; i++){
    cog_straight[i] = (_curvedCoords.at(0).at(0).at(i) + _curvedCoords.at(0).at(cols()-1).at(i) +
		       _curvedCoords.at(rows()-1).at(0).at(i) + _curvedCoords.at(rows()-1).at(cols()-1).at(i))/4;
  }

  for(uint i=0; i<3; i++){
    this->setX(i, _curvedCoords.at(0).at(0).at(i) - cog_straight[i] + cog_curve[i]);
    this->setX(i + 3, _curvedCoords.at(0).at((cols()-1)).at(i) - cog_straight[i] + cog_curve[i]);
    this->setX(i + 6, _curvedCoords.at(rows()-1).at(0).at(i) - cog_straight[i] + cog_curve[i]);
  }
  QuadrilateralSingularity<T>::init();

  determineQuadrilateral();
}

/**
 * ======================================================
 * NODE LOADING
 * ======================================================
 **/


template<typename T>
void Singularity<T>::load(ConfigurationTree conf){
  this->_pi = conf.get<float>("pi", 0.0001);
  this->_sigma = conf.get<float>("sigma", 2);
  this->_maxPi = conf.get<float>("max_pi", 1);
}

template<typename T>
void PointSingularity<T>::load(ConfigurationTree conf){
  Singularity<T>::load(conf);
  conf.requireFields({"point"});
  vector<float> point = conf.get<vector<float>>("point");
  for(uint i=0; i<3; i++){
    X[i] = point.at(i);
  }
}

template<typename T>
void LineSingularity<T>::load(ConfigurationTree conf){
  Singularity<T>::load(conf);
  conf.requireFields({"points"});
  vector<vector<float>> points = conf.get<vector<vector<float> > >("points");
  for(uint i=0; i<2; i++){
    for(uint j=0; j<3; j++){
      X[i*3+j] = points.at(i).at(j);
    }
  }

  lambda = conf.get<float>("lambda", 0);
}


template<typename T>
void CurveSingularity<T>::load(ConfigurationTree conf){
  Singularity<T>::load(conf);
  conf.requireFields({"points"});

  _maxIterations = conf.get<int>("maxits", 0);
  vector<vector<float>> points = conf.get<vector<vector<float> > >("points");
  for(uint i=0; i<points.size(); i++){
    for(uint j=0; j<3; j++){
      _coords.push_back(points.at(i).at(j));
    }
  }
  init();

  LineSingularity<T>::lambda = conf.get<float>("lambda", 0);

  _cacheKey = conf.get<string>("cache_key", "");
}

template<typename T>
void QuadrilateralSingularity<T>::load(ConfigurationTree conf){
  Singularity<T>::load(conf);

  conf.requireFields({"points"});
  vector<vector<float>> points = conf.get<vector<vector<float> > >("points");
  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      X[i*3+j] = points.at(i).at(j);
    }
  }
  init();

  lambda = conf.get<float>("lambda", 0);

}

template<typename T>
void CurvedQuadrilateralSingularity<T>::load(ConfigurationTree conf){
  Singularity<T>::load(conf);

  conf.requireFields({"points"});
  _curvedCoords = conf.get<vector<vector<vector<float> > > >("points");
  _straightCoords = _curvedCoords; // Lazy way to initialize dimensions..
  _maxIterations = conf.get<int>("maxits", 0);

  _cacheKey = conf.get<string>("cache_key", "");

  float shift = conf.get<float>("shift", 0);
  float scale = conf.get<float>("scale", 1);
  
  for(uint i=0; i<_curvedCoords.size(); i++){
    for(uint j=0; j<_curvedCoords.at(0).size(); j++){
      for(uint k=0; k<3; k++){
        _curvedCoords.at(i).at(j).at(k) = (_curvedCoords.at(i).at(j).at(k)+shift)*scale;
      }
    }
  }

  init();

  QuadrilateralSingularity<T>::lambda = conf.get<float>("lambda", 0);
}

