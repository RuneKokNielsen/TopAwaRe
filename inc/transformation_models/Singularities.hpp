#ifndef SINGULARITIES_HPP
#define SINGULARITIES_HPP
#include "Image.hpp"
#include "Node.hpp"

static bool ALLOW_MOVEMENT = false;

/**
 * Common ancestor for all types of singularities.
 **/
template<typename T>
class Singularity: Conf::Node{
public:


  Singularity(){};
  Singularity(float pi, float sigma): _pi(pi), _sigma(sigma) {};
  Singularity(const Singularity<T> *s){ _pi = s->_pi; _sigma = s->_sigma;};

  // The number of optimizable parameters. Currently we only optimize for magnitude (pi)
  virtual uint optsize() const = 0;

  // Packs all member variables into a vector
  virtual vector<float> toVec() const = 0;

  /**
   * Updates optimizable parameters to new values.
   * \param x New values.
   **/
  virtual void update(vector<float> x) = 0;

  virtual float getPi() const{ return _pi; };
  virtual float getSigma() const { return _sigma; };
  float getMaxPi() const{ return _maxPi; };

  virtual Singularity* clone() const = 0;

  /**
   * Updates an alpha field based on the effect of this singularity.
   * \param palpha Pointer to alpha field data.
   * \param X The current transformation.
   * \param meta The image meta of the image, e.g. the alpha field.
   * \param alphaBoundarySmoothing Determines smoothness of alpha field near boundaries of the singularity.
   **/
  virtual void fillAlpha(T* palpha, T** X, ImageMeta meta, float alphaBoundarySmoothing) const = 0;

  /**
   * Updates a transformation by adding the displacement caused by this singularity.
   * \param XW Current transformation to be updated.
   * \param X Current transformation (not updated!)
   * \param meta Meta of the images.
   **/
  virtual void fillWalpha(T** XW, T** X, ImageMeta meta) const = 0;

  /**
   * Updates an inverse transformation by adding the inverse displacement caused
   * by this singularity.
   * \param XW current transformation to be updated.
   * \param X current transformation (not updated!)
   * \param meta Meta of the images.
   **/
  virtual void fillWalphaInv(T** XW, T** X, ImageMeta meta) const = 0;


  virtual ~Singularity(){};

  vector<float> toOptVec() const{
    vector<float> v;
    vector<float> v0 = toVec();
    for(uint i=0; i<this->optsize(); i++){
      v.push_back(v0.at(i));
    }
    return v;
  }

  // Current displacement magnitude (optimized)
  float _pi;
  // Displacement smoothing (configured)
  float _sigma;
  // Maximum displacement magnitude (configured)
  float _maxPi;

  void load(ConfigurationTree conf);

  // True if this is a curved singularity
  virtual bool isCurved(){ return false; };
};


template<typename T>
inline std::ostream &operator<<(std::ostream &os, Singularity<T> const &s) {
  vector<float> v = s.toVec();
  for(uint i=0; i<v.size(); i++){
    os << v.at(i) << " ";
  }
  os << "\n";
  return os;
}

/**
 * Point singularity representing growth from an infinitesimal point
 */
template<typename T>
struct PointSingularity : public Singularity<T>{
  float X[3];
  PointSingularity(float r, float c, float z, float pi, float sigma): Singularity<T>(pi, sigma)
  {
    X[0] = r;
    X[1] = c;
    X[2] = z;
  };

  PointSingularity(const PointSingularity<T>* s): Singularity<T>(s){
    for(uint i=0; i<3; i++){
      X[i] = s->X[i];
    }
  }


  PointSingularity():Singularity<T>(0,0){};
  void load(ConfigurationTree conf);

  uint size() const {
    return 5;
  }

  uint optsize() const {
    return ALLOW_MOVEMENT ? size() : 1;
  }

  vector<float> toVec() const{
    vector<float> v;
    v.push_back(this->_pi);
    v.push_back(this->_sigma);
    v.push_back(X[0]);
    v.push_back(X[1]);
    v.push_back(X[2]);
    return v;
  }

  void update(vector<float> x){
    this->_pi = max(.0f, x.at(0));
    //this->_sigma = x.at(1);
    if(ALLOW_MOVEMENT){
      X[0] = x.at(2);
      X[1] = x.at(3);
      X[2] = x.at(4);
    }
  }

  float getX(int dim) const{
    return X[dim];
  }

  void setX(int dim, float x){
    X[dim] = x;
  }

  PointSingularity* clone() const{
    return new PointSingularity({X[0], X[1], X[2], this->_pi, this->_sigma});
  }

  void fillAlpha(T* palpha, T** X, ImageMeta meta, float abs) const;

  void fillWalpha(T** XW, T** X, ImageMeta meta) const;

  void fillWalphaInv(T** XW, T** X, ImageMeta meta) const;

};



/**
 * Line singularity representing growth from a straight line segment between two points
 */
template<typename T>
struct LineSingularity : public Singularity<T>{
  float X[6];
  float lambda = 0;

  LineSingularity(float r1, float c1, float z1, float r2, float c2, float z2, float pi, float sigma): Singularity<T>(pi, sigma){
    X[0] = r1;
    X[1] = c1;
    X[2] = z1;
    X[3] = r2;
    X[4] = c2;
    X[5] = z2;
  }

  LineSingularity(const LineSingularity<T> *s): Singularity<T>(s){
    for(int i=0; i<6; i++){
      X[i] = s->X[i];
    }
  }

  LineSingularity(): Singularity<T>(0,0){};
  void load(ConfigurationTree conf);

  uint size() const{
    return 8;
  }

  uint optsize() const{
    return ALLOW_MOVEMENT ? size() : 1;
  }

  float getX(int dim) const{
    return X[dim];
  }

  void setX(int dim, float x){
    X[dim] = x;
  }

  vector<float> toVec() const{
    vector<float> v;
    v.push_back(this->_pi);
    v.push_back(this->_sigma);
    for(int i=0; i<6; i++){
      v.push_back(X[i]);
    }
    return v;
  }

  void update(vector<float> x){
    this->_pi = max(.0f, x.at(0));
    //this->_sigma = x.at(1);
    if(ALLOW_MOVEMENT){
      for(int i=0; i<6; i++){
	X[i] = x.at(i+2);
      }
    }
  }

  LineSingularity* clone() const{
    return new LineSingularity<T>(this);
  }

  void fillAlpha(T* palpha, T** X, ImageMeta meta, float abs) const;

  void fillWalpha(T** XW, T** X, ImageMeta meta) const;

  void fillWalphaInv(T** XW, T** X, ImageMeta meta) const;
};

/**
 * @Interface Curved: A curved singularity that is preregistered to a straight
 * one in the same diffeomorphic group.
 * Such a singularity must describe a number of corresponding points in
 * curved space and straight space.
 **/
class CurvedSingularity{
public:

  virtual ~CurvedSingularity(){};
  
  virtual vector<Image> curvedToMesh() = 0;
  virtual vector<Image> straightToMesh() = 0;
  
  virtual void updateCurvedCoords(Image mesh, int dim) = 0;
  virtual void updateStraightCoords(Image mesh, int dim) = 0;

  virtual unsigned int nCoords() = 0;
  virtual float getCurvedCoord(int i, int dim) = 0;
  virtual float getStraightCoord(int i, int dim) = 0;

  virtual CurvedSingularity* clone() const = 0;

  int maxIterations(){ return _maxIterations; };


  CurvedSingularity(){
  };
  CurvedSingularity(const CurvedSingularity* cs){
    _psi = cs->_psi;
    _psiInv = cs->_psiInv;
  }
  
  vector<Image> _psi;
  vector<Image> _psiInv;
  int _maxIterations;

  bool doCache() const { return _cacheKey != ""; };
  string _cacheKey;
};


/**
 * A curve singularity consists of a number of control points.
 * These points are registered onto a straight line, and the
 * growth is then modeled as a straight line singularity in
 * the straightened domain
 **/
template<typename T>
struct CurveSingularity : public LineSingularity<T>, public CurvedSingularity{

  vector<float> _coords;
  vector<float> _lineCoords;

  CurveSingularity(){};
  void load(ConfigurationTree conf);
  CurveSingularity(vector<float> coords, float pi, float sigma): LineSingularity<T>(0,0,0,0,0,0,pi, sigma), _coords(coords){
    init();
  }

  void init(){
    determineLine();
    // Set endpoints of straight line singularity
    for(int i=0; i<3; i++){
      this->setX(i, _lineCoords.at(i));
      this->setX(i + 3, _lineCoords.at(_lineCoords.size() - 3 + i));    
    }

  }


  CurveSingularity(const CurveSingularity* cs): LineSingularity<T>(cs), CurvedSingularity(cs){
    _coords = cs->_coords;
    _lineCoords = cs->_lineCoords;
  }

  unsigned int nCoords(){
    return _coords.size() / 3;
  }

  /**
   * Set the endpoints of the straight line segment to be approximated.
   * These endpoints are chosen as the projections of the curve endpoints
   * onto the closest straight line, found through linear regression.
   **/
  void determineLine();

  void projectToLine(float* p0, float* p1, float* x, float* xl);

  void projectToLine(float* w, float* x, float* xl);

  vector<Image> curvedToMesh();

  vector<Image> straightToMesh();

  void updateCurvedCoords(Image mesh, int dim);
  void updateStraightCoords(Image mesh, int dim);

  vector<Image> curveToMesh(vector<float> curve);

  float getCurvedCoord(int i, int dim);
  float getStraightCoord(int i, int dim);

  CurveSingularity* clone() const{
    return new CurveSingularity<T>(this);
  }

  bool isCurved(){ return true; }
};

/**
 * Quadrilateral singularity representing growth from a straight quadrilateral
 */
template<typename T>
struct QuadrilateralSingularity : public Singularity<T>{

  float X[9];
  // The quadrilateral plane is described by a basis B = [p0p1, p0p2, normal/|normal|]
  float B[3][3];
  float Binv[3][3];
  float lambda = 0;

  QuadrilateralSingularity(const QuadrilateralSingularity<T> *s): Singularity<T>(s){
    for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
	X[i*3+j] = s->X[i*3+j];
	B[i][j] = s->B[i][j];
	Binv[i][j] = s->Binv[i][j];
      }
    }
  }

  QuadrilateralSingularity(): Singularity<T>(0, 0){};
  void load(ConfigurationTree conf);

  void init();

  void determineBasis();

  QuadrilateralSingularity* clone() const{
    return new QuadrilateralSingularity<T>(this);
  }

  uint size() const{
    return 11;
  }

  uint optsize() const{
    return ALLOW_MOVEMENT ? size():1;
  }

  float getX(int dim) const{
    return X[dim];
  }

  void setX(int dim, float x){
    X[dim] = x;
  }

  vector<float> toVec() const{
    vector<float> v;
    v.push_back(this->_pi);
    v.push_back(this->_sigma);
    for(int i=0; i<9; i++){
      v.push_back(X[i]);
    }
    return v;
  }



  void update(vector<float> x){
    this->_pi = max(.0f, x.at(0));
    //this->_sigma = x.at(1);
    if(ALLOW_MOVEMENT){
      for(int i=0; i<9; i++){
	X[i] = x.at(i+1);
      }
    }
  }

  void fillAlpha(T* palpha, T** X, ImageMeta meta, float abs) const;

  void fillWalpha(T** XW, T** X, ImageMeta meta) const;

  void fillWalphaInv(T** XW, T** X, ImageMeta meta) const;

};

template<typename T>
struct CurvedQuadrilateralSingularity : public QuadrilateralSingularity<T>, public CurvedSingularity{

  vector<vector<vector<float>>> _curvedCoords;
  vector<vector<vector<float>>> _straightCoords;

  CurvedQuadrilateralSingularity(){};
  void load(ConfigurationTree conf);

  void init();

  CurvedQuadrilateralSingularity(const CurvedQuadrilateralSingularity* cs): QuadrilateralSingularity<T>(cs){
    _curvedCoords = cs->_curvedCoords;
    _straightCoords = cs->_straightCoords;
    _psi = cs->_psi;
    _psiInv = cs->_psiInv;
  }

  unsigned int rows(){
    return _curvedCoords.size();
  }

  unsigned int cols(){
    return _curvedCoords.at(0).size();
  }

  unsigned int nCoords(){
    return rows() * cols();
  }

  void determineQuadrilateral();

  CurvedQuadrilateralSingularity* clone() const{
    return new CurvedQuadrilateralSingularity<T>(this);
  }

  vector<Image> curvedToMesh();
  vector<Image> straightToMesh();
  vector<Image> curveToMesh(vector<vector<vector<float>>> curve);
  void updateCurvedCoords(Image mesh, int dim);
  void updateStraightCoords(Image mesh, int dim);
  float getCurvedCoord(int i, int dim);
  float getStraightCoord(int i, int dim);

  bool isCurved(){ return true; }
};

template class PointSingularity<float>;
template class LineSingularity<float>;
template class CurveSingularity<float>;
template class QuadrilateralSingularity<float>;
template class CurvedQuadrilateralSingularity<float>;
#endif
