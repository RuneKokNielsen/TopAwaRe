#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <chrono>


#ifdef CUDA_GPU
#include "gpu.h"
#endif

using namespace std;
typedef std::chrono::high_resolution_clock Clock;


typedef uint uint;

namespace Imaging{

  /**
   * Complex number struct consisting of two floats (real + imaginary parts).
   **/
  struct complex{
    float r;
    float i;
    complex(double _r) { r = _r; i = _r;};
    complex(double _r, double _i) { r = _r; i = _i;  };
    complex(int n) { r = n; i=0; };
    complex() { r = 0; i = 0; };
    operator double() { return (double) r; };
    complex operator+=(const complex o){
      r = r + o.r;
      i = i + o.i;
      return complex(r, i);
    }
  };


  enum ImageDataType { uint8, f32, f64, cpx };
  template<typename T> ImageDataType typeToImgType();
  template<> ImageDataType typeToImgType<uint8_t>();
  template<> ImageDataType typeToImgType<float>();
  template<> ImageDataType typeToImgType<double>();
  template<> ImageDataType typeToImgType<complex>();

  /**
   * Describe a 3D cube of voxel indeces.
   **/
  struct Cube{
    int r0, r1, c0, c1, z0, z1;
    Cube(int _r0, int _r1, int _c0, int _c1, int _z0, int _z1) { r0 = _r0; r1 = _r1; c0 = _c0; c1 = _c1; z0 = _z0; z1 = _z1; };

    // Constructs the minimal cube containing both cubes
    Cube(Cube a, Cube b){
      r0 = min(a.r0, b.r0);
      r1 = max(a.r1, b.r1);
      c0 = min(a.c0, b.c0);
      c1 = max(a.c1, b.c1);
      z0 = min(a.z0, b.z0);
      z1 = max(a.z1, b.z1);
    }

    Cube(vector<int> v){
      r0 = v.at(0);
      r1 = v.at(1);
      c0 = v.at(2);
      c1 = v.at(3);
      z0 = v.at(4);
      z1 = v.at(5);
    }
  };


#ifdef CUDA_GPU
  /**
   * enum used in the image meta to tell whether the
   * intensity data is stored in regular memory for
   * cpu processing or in the gpu device's memory
   * for gpu processing.
   **/
  enum mode {cpu, gpu, error};
#endif

  /**
   * Meta information describing an image in terms of e.g. dimensions.
   **/
  struct ImageMeta {
    uint height = 0;
    uint width = 0;
    uint depth = 0;
    uint dimensions = 0;
    ImageDataType dtype = f32;
    uint dsize = 4;
    uint ncells = 0;
    uint nbytes = 0;
#ifdef CUDA_GPU
    mode m = cpu;
#endif

    ImageMeta() = default;
    ImageMeta(uint _height, uint _width, uint _depth, uint _dimensions, ImageDataType _dtype, uint _dsize) :
      height(_height), width(_width), depth(_depth), dimensions(_dimensions), dtype(_dtype), dsize(_dsize),
      ncells(_height*_width*_depth), nbytes(dsize*ncells){};

    std::vector<int> dims() const{
      std::vector<int> vdims;
      vdims.push_back(height);
      vdims.push_back(width);
      vdims.push_back(depth);
      return vdims;
    }

    bool operator==(const ImageMeta o){
      return height == o.height
        && width == o.width
        && depth == o.depth
        && dtype == o.dtype;
    }

    /**
     * Get the length in voxels of the i'th dimension.
     **/
    int dim(int i) const{
      switch(i){
      case 0:
        return height;
      case 1:
        return width;
      case 2:
        return depth;
      };
      return -1;
    };

    string to_string() const{
      return std::to_string(height) + "x" + std::to_string(width) + "x" + std::to_string(depth);
    }
  };

  /**
   * When padding an image, this enum signifies either a padding of
   * all zeros or a mirroring of intensities over the boundary.
   **/
  enum PadType { zero, mirror };


  /**
   * The Image class wraps a one-dimensional data array image representation.
   * The array is internally stored as a char array and is operated by
   * retrieving a pointer to the start.
   *
   * This wrapper seeks to remove the requirement for potentially confusing use
   * pointers in the remaining implementation. This is achieved by wrapping the
   * data array in an shared pointer object which is then shared when copying
   * the object and deleted automatically once all references to it are out of scope.
   * This effectively means that altering direct copies of an image object will
   * also alter the original image object. If a deep copy is needed use the clone
   * function.
   */
  class Image{

  public:

    /**
     * Creates a new image based on a meta object.
     * \param meta The meta of the new image.
     * \param zerofill If true, sets all voxels to 0.
     **/
    Image(ImageMeta meta, bool zerofill=false): _meta(meta){
      _meta.ncells = meta.height * meta.width * meta.depth;
      _meta.dsize = getdsize(_meta.dtype);
      _meta.nbytes = _meta.ncells * _meta.dsize;
      _meta.dimensions = meta.depth > 1 ? 3 : 2;

#ifdef CUDA_GPU
      if(_meta.m == gpu){
        initGData(zerofill);
        return;
      }
#endif
      initData(zerofill);
    }

    // Empty image constructor
    Image(){};

    ~Image(){
    }

    /**
     * Shallow copy an image.
     * \param obj The image to shallow copy.
     **/
    Image(const Image &obj){
      _meta = obj._meta;    
      _data = obj._data;

#ifdef CUDA_GPU
      _gdata = obj._gdata;
#endif
    }

    /**
     * Setting an image equal to another corresponds
     * to a shallow copy.
     * \param obj Other image.
     **/
    Image& operator=(Image obj){
      _meta = obj._meta;
      _data = obj._data;

#ifdef CUDA_GPU
      _gdata = obj._gdata;
#endif
      return *this;
    }

    /**
     * Get the data size in bytes of a data type.
     * \param dtype The data type to query.
     **/
    uint getdsize(ImageDataType dtype){
      switch(dtype){
      case f32:
        return sizeof(float);
      case f64:
        return sizeof(double);
      case uint8:
        return sizeof(uint8_t);
      case cpx:
        return sizeof(complex);
      }
      return 1;
    }

    /**
     * Returns a new image taken from a subcube of this image.
     **/
    template<typename T> Image slice(int r0, int r1, int c0, int c1, int z0, int z1) const{
      ImageMeta meta = _meta;
      meta.height = r1 - r0;
      meta.width = c1 - c0;
      meta.depth = z1 - z0;


      Image im = Image(meta, false);

#ifdef CUDA_GPU
      if(meta.m == gpu){
        cuda_copy_slice(im.meta().ncells, gptr(), im.gptr(),_meta.height, _meta.width, _meta.depth, meta.height, meta.width, meta.depth, r0, c0, z0);
        return im;
      }
#endif

      T *ptrSrc = ptr<T>();
      T *ptrDest = im.ptr<T>();

      int nperrow_dest = meta.depth * meta.width;
      int nperrow_src = _meta.depth * _meta.width;
#pragma omp parallel for
      for(uint i=0; i<im.meta().ncells; i++){

        int rdest = i / nperrow_dest;
        int rsrc = rdest + r0;

        int cdest = (i % nperrow_dest) / meta.depth;
        int csrc = cdest + c0;

        int zdest = i % meta.depth;
        int zsrc = zdest + z0;

        int j = rsrc * nperrow_src + csrc * _meta.depth + zsrc;
        ptrDest[i] = ptrSrc[j];
      }

      return im;
    }

    /**
     * See slice(int, int, int, int, int, int)
     **/
    template<typename T> Image slice(Cube bb) const{
      return slice<T>(bb.r0, bb.r1, bb.c0, bb.c1, bb.z0, bb.z1);
    }

    /**
     * Returns a deep copy of this image
     */
    Image clone() const{
      Image im(_meta);
      copy_to(im);
      return im;
    }

    /**
     * Set the value at a given coordinates. Only works in CPU memory.
     **/
    template<typename T>
    void set(int r, int c, int z, T val){
      ptr<T>()[r * meta().dim(1) * meta().dim(2) + c * meta().dim(2) + z] = val;
    }


    /**
     * Get the value at a given coordinate. Only works in CPU memory.
     **/
    template<typename T>
    T get(int r, int c, int z){
      return ptr<T>()[r * meta().dim(1) * meta().dim(2) + c * meta().dim(2) + z];
    }

    /**
     * Copy the data from this image to the data at another image.
     **/
    void copy_to(Image target) const{

#ifdef CUDA_GPU
      const Image *imgs[2] = {this, &target};
      if(modeCheck(imgs, 2) == gpu){
        cuda_copy(_meta.ncells * (_meta.dtype == cpx ? 2 : 1), gptr(), target.gptr());
        return;
      }
#endif

      char *a = ptr<char>();
      char *b = target.ptr<char>();
#pragma omp for
      for(uint i=0; i<_meta.nbytes; i++){
        b[i] = a[i];
      }
    }


    uint height() const { return _meta.height; };
    uint width() const { return _meta.width; };
    uint depth() const { return _meta.depth; };
    uint dimensions() const { return _meta.dimensions; };
    ImageDataType type() const { return _meta.dtype; };

    ImageMeta meta() const { return _meta; };

    // Returns a pointer to the start of the data. The data type
    // must be explicitly given at compilation time.
    template<typename T> T* ptr() const {
#ifdef CUDA_GPU
      if(_meta.m != cpu){
        throw invalid_argument("RAM pointer requested but image not in RAM!");
      }
#endif
      return (T*) _data.get();
    }



#ifdef CUDA_GPU

    /**
     * Return the memory mode (cpu or gpu) of this image.
     **/
    mode getMode() const{
      return _meta.m;
    }

    /**
     * Takes an array of pointers to images and returns their
     * memory mode if they are all the same or throws an exception
     * if they are not.
     **/
    static mode modeCheck(const Image **imgs, int n){
      if(n < 1) return mode::error;
      mode m = imgs[0]->getMode();
      for(int i=1; i<n; i++){
        if(m != imgs[i]->getMode()){
          // Modes must always match!
          string modes;
          for(int j=0; j<n; j++){
            modes += imgs[j]->getMode() + " ";
          }
          throw logic_error("Memory modes do not match! (" + modes + ")");
        }
      }
      return m;
    }

    /**
     * Switches the memory mode of this image. If it is in cpu
     * it moves data to the gpu and vice-versa.
     **/
    void modeSwitch(){
      switch(_meta.m){
      case cpu:
        toGPU();
        break;
      case gpu:
        toCPU();
        break;
      case error:
        throw logic_error("This should not occur!");
      }
    }

    // If in cpu mode -- allocates vram and copies from ram
    void toGPU(){
      if(_meta.m != cpu){
        cout << "Warning: calling toGPU on image already on GPU!\n";
        return;
      }
      _toGPU();
    }

    // If in gpu mode -- copies data from vram to ram
    void toCPU(){
      if(_meta.m != gpu){
        cout << "Warning: calling toCPU on image already on CPU!\n";
        return;
      }
      _toCPU();
    }

    /**
     * Returns a pointer to the intensity data in gpu memory
     * assuming that it resides on the gpu.
     **/
    float *gptr() const{
      if(_meta.m != gpu){
        throw invalid_argument("VRAM pointer requested but image not in VRAM!");
      }
      return _gdata.get();
    }
#endif

  private:

#ifdef CUDA_GPU
    /**
     * Pointer to the intensity data in gpu.
     * The intensity data is stored in a shared pointer.
     * This allows us to shallow copy images sharing the
     * intensity data across multiple objects without
     * worrying about when to finally free the memory.
     **/
    shared_ptr<float> _gdata;

    struct gDel {
      /**
       * Deleter-class for the shared_ptr to vram.
       * When reference count is zero the vram is freed.
       **/
      void operator()(float* p) const {
        cuda_f32_free(p);
      }
    };

    /**
     * Moves the intensity data of this image from cpu
     * memory to the gpu memory.
     **/
    void _toGPU(){
      initGData(false);
      cuda_f32_send(_meta.ncells * (_meta.dtype == cpx ? 2 : 1), (float*) _data.get(), gptr());
      _data.reset();

    }

    /**
     * Moves the intensity data of this image from gpu
     * memory to regular memory.
     **/
    void _toCPU(){
      initData(false);
      cuda_f32_retrieve(_meta.ncells * (_meta.dtype == cpx ? 2 : 1), ptr<float>(), _gdata.get());
      _gdata.reset();
      _meta.m = cpu;
    }

    /**
     * Allocates and initializes intensity data
     * in gpu memory.
     **/
    void initGData(bool zerofill=false){
      int n = _meta.ncells;
      if( _meta.dtype == cpx ) n <<= 1;
      float *gdata = cuda_f32_allocate(n, zerofill);
      _gdata = shared_ptr<float>(gdata, gDel());
      _meta.m = gpu;
    }

#endif
    /**
     * Allocates and initializes intensity data in
     * regular memory.
     **/
    void initData(bool zerofill=false){
      // Zero-initialization
      if(zerofill){
        _data = shared_ptr<char> (new char[_meta.nbytes](),
                                  default_delete<char[]>());
      }else{
        _data = shared_ptr<char> (new char[_meta.nbytes],
                                  default_delete<char[]>());
      }
#ifdef CUDA_GPU
      _meta.m = cpu;
#endif
    }

    /**
     * The meta information about this image object.
     **/
    ImageMeta _meta;
    /**
     * Pointer to the intensity data in regular memory.
     * The intensity data is stored in a shared pointer.
     * This allows us to shallow copy images sharing the
     * intensity data across multiple objects without
     * worrying about when to finally free the memory.
     **/
    shared_ptr<char> _data;


  };

}
using namespace Imaging;


/**
 * ===================================================
 *               Operators
 * ===================================================
 */

Image operator*(const Image &A, const double s);
Image operator*(const double s, const Image &A);
Image operator*(const Image &A, const Imaging::complex s);
Image operator*(const Imaging::complex s, const Image &A);
Image operator*(const Image &A, const Image &B);
Image operator*(const Image &A, const float s);
Image operator+(const Image &A, const double s);
Image operator+(const Image &A, const float s);
Image operator+(const double s, const Image &A);
Image operator+(const Image &A, const Imaging::complex s);
Image operator+(const Imaging::complex s, const Image &A);
Image operator+(const Image &A, const Image &B);
Image operator-(const Image &A, const Image &B);
ostream& operator<< ( ostream& outs, const Image &A );
ostream& operator<< ( ostream& outs, const ImageMeta &A );

/**
 * Helper for various common image operators.
 **/
class ImageUtils{
public:

  /**
   * Element-wise multiplication of images.
   * \param A One image
   * \param B Another image
   * \param target The resulting image A .* B.
   **/
  static void multiply(const Image A, const Image B, Image target);
  /**
   * Element-wise division of images.
   * \param A One image.
   * \param B Another image.
   * \param target The resulting image A ./ B.
   **/
  static void divide(const Image A, const Image B, Image target);
  /**
   * Element-wise subtraction of images.
   * \param A One image.
   * \param B Another image.
   * \param target The resulting image A - B.
   **/
  static void subtract(const Image A, const Image B, Image target);
  /**
   * Element-wise addition of images.
   * \param A One image.
   * \param B Another image.
   * \param target The resulting image A + B.
   **/
  static void m_add(const Image A, const Image B, Image target);

  /**
   * Scaling of image intensities.
   * \param A An image to scale.
   * \param s The scalar.
   * \param target The resulting image s * A
   **/
  template <typename T> static void scalar(const Image A, T s, Image target);

  /**
   * Scalar addition of scalar to all voxels in an image.
   * \param A An image.
   * \param s The scalar.
   * \param target The resulting image s + A.
   **/
  template <typename T> static void s_add(const Image A, T s, Image target);
  /**
   * Compute the Jacobian determinants of a transformation.
   * \param f 2D or 3D transformation.
   **/
  static Image jacobianDeterminants(const vector<Image> f);
  /**
   * Compute the finite difference image gradient in one direction.
   * \param A The image whose gradient to compute.
   * \param direction The direction/dimension to compute the gradient for.
   **/
  static Image gradient(const Image A, int direction);
  /**
   * Compute the finite difference image gradients in all directions.
   * \param A The image whose gradients to compute.
   **/
  static vector<Image> gradients(const Image A);
  /**
   * Compute the fast Fourier transformation of an image.
   * \param A The image to transform to Fourier domain.
   **/
  static Image fft(const Image A);
  /**
   * Compute the inverse fast Fourier transformation of an
   * image in Fourier domain.
   * \param A The image to inverse.
   * \param type The data type of the image in spatial domain.
   **/
  static Image ifft(const Image A, ImageDataType type);
  /**
   * Compute zero-frequency shift of an image in Fourier domain.
   * \param A The image to shift.
   **/
  static Image fftShift(const Image A);
  /**
   * Returns a string-representation of the image intensities.
   **/
  static string toString(const Image A);
  /**
   * Compute a meshgrid with the specified meta.
   * \param meta The meta of the grid images.
   **/
  static vector<Image> meshgrid(const ImageMeta meta);
  /**
   * Pad an image with extra voxels on all sides.
   * \param A The image to pad.
   * \param width The padding at each side.
   * \param type The type of padding.
   **/
  static Image pad(const Image A, int width, PadType type=mirror);
  /**
   * Unpads an image by removing the first and last voxels in all dimensions.
   * \param A The image to unpad.
   * \param width The number of voxels to remove on each side.
   **/
  static Image unpad(const Image A, int width);
  /**
   * Displays an image and halts for some time.
   * If the image is 3D, shows each slice one at a time.
   * \param A The image to show.
   * \param ms The time in ms to halt. If A is 3D, this is the time for each slice.
   **/
  static void showImage(const Image A, int ms = 1);

  /**
   * Normalizes an image to [0, 1] interval in-place.
   * \param A The image to normalize.
   **/
  static void normalize(Image A);

  /**
   * Returns a slice of an image defined by a cube.
   * \param A The image to slice.
   * \param cube The cube to cut out.
   **/
  static Image slice(const Image A, Cube cube);

  /**
   * Sums over the voxel values in an image.
   * \param A The image to sum over.
   **/
  static double sum(const Image A);

  /**
   * Filters an image with a filter in the Fourier domain.
   * \param A An image in spatial domain to be filtered.
   * \param f A filter in Fourier domain to filter with.
   **/
  static Image filter(const Image A, const Image f);

  /**
   * Applies an alpha channel to an image and sets the background to
   * a constant value beta. The new image is given by A.*alpha + (1-alpha)*beta
   * \param A The source image.
   * \param alpha The alpha channel.
   * \param B The new image.
   * \param beta The background intensity.
   **/
  static void betaImage(const Image A, Image alpha, Image B, float beta);
};


/**
 * Templated backend for ImageUtils. See function descriptions in ImageUtils.
 **/
template <typename T>
class _ImageUtils{

public:
  static void multiply(const Image A, const Image B, Image target);
  static void divide(const Image A, const Image B, Image target);
  static void scalar(const Image A, T s, Image target);
  static void s_add(const Image A, T s, Image target);
  static void subtract(const Image A, const Image B, Image target);
  static void m_add(const Image A, const Image B, Image target);
  static Image jacobianDeterminants(const vector<Image> f);
  static Image gradient(const Image A, int direction);
  static vector<Image> gradients(const Image A);
  static Image fft(const Image A);
  static Image ifft(const Image A);
  static Image fftShift(const Image A);
  static Imaging::complex toComplex(const T a);
  static T fromComplex(const Imaging::complex a);
  static string toString(const Image A);
  static ImageDataType dtype();
  static vector<Image> meshgrid(const ImageMeta meta);
  static Image pad(const Image A, int width, PadType type);
  static Image unpad(const Image A, int width);
  static void showImage(const Image A, int ms);
  static void normalize(Image A);
  static Image slice(const Image A, Cube cube);
  static double sum(const Image A);
  static Image filter(const Image A, const Image f);
};


/**
 * Complex structure used for frequency domain
 */
Imaging::complex operator*(Imaging::complex a, Imaging::complex b);
Imaging::complex operator*(Imaging::complex a, double s);
Imaging::complex operator*(int a, Imaging::complex b);
Imaging::complex operator-(Imaging::complex a, Imaging::complex b);
Imaging::complex operator/(Imaging::complex a, double s);
Imaging::complex operator/(Imaging::complex a, int s);
Imaging::complex operator/(Imaging::complex a, Imaging::complex b);
Imaging::complex operator+(Imaging::complex a, Imaging::complex b);
bool operator>(Imaging::complex a, Imaging::complex b);
ostream& operator<< ( ostream& outs, Imaging::complex a);
ostream& operator<< ( ostream& outs, Cube cube);

template <typename T>
ostream& operator<< ( ostream& outs, vector<T> v);

#define IM_ARITH_B(t, pre, post) switch(t){                   \
  case f32: pre _ImageUtils<float>::post; break;              \
  case f64: pre _ImageUtils<double>::post; break;             \
  case uint8: pre _ImageUtils<uint8_t>::post; break;          \
  case cpx: pre _ImageUtils<Imaging::complex>::post; break;   \
  default: throw logic_error("This should not happen!");      \
 };
#define IM_ARITH_A(t, f) IM_ARITH_B(t, , f);

class Utils{
private:
  static chrono::_V2::system_clock::time_point _tic;

public:
  static chrono::_V2::system_clock::time_point  tic();

  static std::chrono::microseconds toc(chrono::_V2::system_clock::time_point  t);

  static std::chrono::microseconds toc();

  static void showImage(const Image A, int ms = 1);
};

#define TIC Utils::tic();
#define TOC Utils::toc();
#define tTOC(t) Utils::toc(t);
#define SHOW(I) Utils::showImage(I, 1);
#define SHOWWAIT(I, ms) Utils::showImage(I, ms);
#define SHOWPAUSE(I) Utils::showImage(I, 0);


struct NotImplementedException: public std::exception{  
  std::string _str;
  NotImplementedException() : _str() {};
  NotImplementedException(std::string str) : _str(str) {}
  ~NotImplementedException() throw () {} 
  const char* what() const throw() { return _str.c_str(); }
};

#define NOT_IMPLEMENTED throw NotImplementedException(__FUNCTION__);

#endif
