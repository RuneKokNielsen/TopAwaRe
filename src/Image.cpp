
#include <Image.hpp>
#include <exception>
#include <iostream>
#include <omp.h>
#include "math.h"

#ifdef CUDA_GPU
#include "gpu.h"
#endif

/**
 * ===================================
 * Data type independent implementations
 * ===================================
 */


template<> ImageDataType Imaging::typeToImgType<uint8_t>(){return uint8;}
template<> ImageDataType Imaging::typeToImgType<float>(){return f32;}
template<> ImageDataType Imaging::typeToImgType<double>(){return f64;}
template<> ImageDataType Imaging::typeToImgType<complex>(){return cpx;}  


/**
 * ===================================
 * Meshgrid generation
 * ===================================
 */

template<typename T> vector<Image> _ImageUtils<T>::meshgrid(const ImageMeta meta){
  T* delta = new T[meta.dimensions];
  int* pos = new int[meta.dimensions];
  T** ptrs = new T*[meta.dimensions];
  uint* dims = new uint[meta.dimensions];

  
  vector<Image> grid;    
  
  for(uint d=0; d<meta.dimensions; d++){
    dims[d] = meta.dims().at(d);
    delta[d] = 1.0 / (dims[d]);
    pos[d] = 0;
    grid.push_back(Image(meta));

#ifdef CUDA_GPU
    if(meta.m == gpu){
      ptrs[d] = (T*) grid.at(d).gptr();
    }else{
#endif
    ptrs[d] = grid.at(d).ptr<T>();
#ifdef CUDA_GPU
    }
#endif
  }


#ifdef CUDA_GPU
  if(meta.m == gpu){
    cuda_meshgrid(meta.ncells, (float**) ptrs, meta.dimensions, dims, (float*) delta);    
  }else{
#endif
  
  for(uint i=0; i<meta.ncells; i++){
    
    uint d = meta.dimensions - 1;
    if(i>0) pos[d]++;
    while(pos[d] == (int) dims[d]){
      pos[d] = 0;
      d--;      
      pos[d]++;
    }
    
    for(uint d=0; d<meta.dimensions; d++){     
      *(ptrs[d]) = (T) pos[d] * delta[d];
      ptrs[d]++;
    }
  }

#ifdef CUDA_GPU
  }
#endif

  delete[] delta;
  delete[] pos;
  delete[] ptrs;
  delete[] dims;
  return grid;
}
vector<Image> ImageUtils::meshgrid(const ImageMeta meta){
  IM_ARITH_B(meta.dtype, return, meshgrid(meta));
}


/**
 * ===================================
 * Element-wise division
 * ===================================
 */

void ImageUtils::divide(const Image A, const Image B, Image target){
  IM_ARITH_A(A.type(), divide(A, B, target));
}
template <typename T>void _ImageUtils<T>::divide(const Image A, const Image B, Image target){

#ifdef CUDA_GPU
  const Image *imgs[3] = {&A, &B, &target};
  if(Image::modeCheck(imgs, 3) == mode::gpu){
    if(A.meta().dtype == cpx){
      cuda_div_cpx(A.meta().ncells, (cufftComplex*) A.gptr(), (cufftComplex*) B.gptr(), (cufftComplex*) target.gptr());
    }else{
      cuda_div(A.meta().ncells, A.gptr(), B.gptr(), target.gptr());
    }
    
    return;
  }
#endif
  
  T *ptrA = A.ptr<T>();
  T *ptrT = target.ptr<T>();
  if(B.type() == f64 && A.type() != f64){
    double *ptrB = B.ptr<double>();
    #pragma omp parallel for
    for(uint i=0; i<A.meta().ncells; i++){
      ptrT[i] = ptrA[i] / ptrB[i];
    }
  }else{
    T *ptrB = B.ptr<T>();
#pragma omp parallel for
    for(uint i=0; i<A.meta().ncells; i++){
      ptrT[i] = ptrA[i] / ptrB[i];
    } 
  }  
    
 
}


/**
 * ===================================
 * Element-wise multiplication
 * ===================================
 */

void ImageUtils::multiply(const Image A, const Image B, Image target){
  IM_ARITH_A(A.type(), multiply(A, B, target));
}
template <typename T>void _ImageUtils<T>::multiply(const Image A, const Image B, Image target){


#ifdef CUDA_GPU
  const Image *imgs[3] = {&A, &B, &target};
  if(Image::modeCheck(imgs, 3) == mode::gpu){
    if(A.meta().dtype == cpx){
      cuda_mult_cpx(A.meta().ncells, (cufftComplex*) A.gptr(), (cufftComplex*) B.gptr(), (cufftComplex*) target.gptr());
    }else{
      cuda_mult(A.meta().ncells, A.gptr(), B.gptr(), target.gptr());
    }
    
    return;
  }
#endif

  T *ptrA = A.ptr<T>();
  T *ptrB = B.ptr<T>();
  T *ptrT = target.ptr<T>();
  #pragma omp parallel for
  for(uint i=0; i<A.meta().ncells; i++){
    ptrT[i] = ptrA[i] * ptrB[i];
  }

}





/**
 * ===================================
 * Scalar products
 * ===================================
 */

template <typename T> void ImageUtils::scalar(Image A, T s, Image target){
  IM_ARITH_A(A.type(), scalar(A, s, target));
  //      _ImageUtils<T>::scalar(A, s, target);
}

template <typename T> void _ImageUtils<T>::scalar(Image A, T s, Image target){


#ifdef CUDA_GPU
  const Image *imgs[2] = {&A, &target};
  if(Image::modeCheck(imgs, 2) == mode::gpu){
    cuda_scalar(A.meta().ncells, A.gptr(), (float) s, target.gptr());
    return;
  }
#endif

  T *ptrA = A.ptr<T>();
  T *ptrT = target.ptr<T>();
#pragma omp parallel for
  for(uint i=0; i<A.meta().ncells; i++){
    ptrT[i] = ptrA[i] * s;
  }
}


/**
 * ===================================
 * Scalar addition
 * ===================================
 */


template <typename T> void ImageUtils::s_add(const Image A, T s, Image target){
  IM_ARITH_A(A.type(), s_add(A, s, target));
  //_ImageUtils<T>::add(A, s, target);
}
template<typename T> void _ImageUtils<T>::s_add(const Image A, T s, Image target){

#ifdef CUDA_GPU
  const Image *imgs[2] = {&A, &target};
  if(Image::modeCheck(imgs, 2) == mode::gpu){
    cuda_sadd(A.meta().ncells, A.gptr(), (float) s, target.gptr());
    return;
  }
#endif
  
  
  T *ptrA = A.ptr<T>();
  T *ptrT = target.ptr<T>();
  
  #pragma omp parallel for
  for(uint i=0; i<A.meta().ncells; i++){
    ptrT[i] = ptrA[i] + s;
  }
}

/**
 * ===================================
 * Element-wise subtraction
 * ===================================
 */

void ImageUtils::subtract(const Image A, const Image B, Image target){
  IM_ARITH_A(A.type(), subtract(A, B, target));
}
template<typename T> void _ImageUtils<T>::subtract(const Image A, const Image B, Image target){

#ifdef CUDA_GPU
  const Image *imgs[3] = {&A, &B, &target};
  if(Image::modeCheck(imgs, 3) == mode::gpu){
    cuda_msub(A.meta().ncells, A.gptr(), B.gptr(), target.gptr());
    return;
  }
#endif
  
  T *ptrA = A.ptr<T>();
  T *ptrB = B.ptr<T>();
  T *ptrT = target.ptr<T>();
#pragma omp parallel for
  for(uint i=0; i<A.meta().ncells; i++){
    ptrT[i] = ptrA[i] - ptrB[i];
  }
}


/**
 * ===================================
 * Element-wise addition
 * ===================================
 */

void ImageUtils::m_add(const Image A, const Image B, Image target){
  IM_ARITH_A(A.type(), m_add(A, B, target));
}
template<typename T> void _ImageUtils<T>::m_add(const Image A, const Image B, Image target){

#ifdef CUDA_GPU
  const Image *imgs[3] = {&A, &B, &target};
  if(Image::modeCheck(imgs, 3) == mode::gpu){
    cuda_madd(A.meta().ncells, A.gptr(), B.gptr(), target.gptr());
    return;
  }
#endif

  T *ptrA = A.ptr<T>();
  T *ptrB = B.ptr<T>();
  T *ptrT = target.ptr<T>();
#pragma omp parallel for
  for(uint i=0; i<A.meta().ncells; i++){
    ptrT[i] = ptrA[i] + ptrB[i];
  }

}



/**
 * ===================================
 * Image normalization
 * ===================================
 */
void ImageUtils::normalize(Image A){
  IM_ARITH_A(A.type(), normalize(A));
}
template<typename T> void _ImageUtils<T>::normalize(Image A){
  
    T *p = A.ptr<T>();
    T minv = numeric_limits<T>::max();
    T maxv = numeric_limits<T>::min();

    for(uint i=0; i<A.meta().ncells; i++){
      if(p[i] < minv) minv = p[i];
      if(p[i] > maxv) maxv = p[i];      
    }

    maxv = maxv - minv;
    #pragma omp parallel for
    for(uint i=0; i<A.meta().ncells; i++){
      p[i] = (p[i] - minv) / maxv;
    }
    
}




/**
 * ===================================
 * Filtering
 * ===================================
 */

Image ImageUtils::filter(const Image A, const Image f){
  IM_ARITH_B(A.type(), return, filter(A, f));
}

template <typename T> Image _ImageUtils<T>::filter(const Image A, const Image f){

  Image FA = fft(A);
  FA = ImageUtils::fftShift(FA);
  FA = FA * f;
  FA = ImageUtils::fftShift(FA);
  return ImageUtils::ifft(FA, f32);

}


/**
 * ====================================
 * Image slicing
 * ====================================
 */

Image ImageUtils::slice(const Image A, Cube cube){
  IM_ARITH_B(A.type(), return, slice(A, cube));
}
template<typename T> Image _ImageUtils<T>::slice(const Image A, Cube cube){
  return A.slice<T>(cube);
}



/**
 * ===================================
 * Gradients computed by central differences with Neumann zero-derivative conditions
 * direction 0 is down in rows, 1 is right in cols and 2 forward in depth
 * ===================================
 */
Image ImageUtils::gradient(const Image A, int direction){
  IM_ARITH_B(A.type(), return, gradient(A, direction));
}
template<typename T> Image _ImageUtils<T>::gradient(const Image A, int direction){
  Image B = Image(A.meta());
  
  T *ptrA = A.ptr<T>();
  T *ptrB = B.ptr<T>();  

  uint n_per_row = A.meta().width * A.meta().depth;
  uint n = A.meta().ncells;

  #pragma omp parallel for
  for(uint i=0; i<n; i++){
    uint l = i;
    uint r = i;
    uint c;
    uint z;
    int cnt = 2;
    switch(direction){
    case 0:
      if(i >= n_per_row) l = i - n_per_row;
      if(i < n - n_per_row) r = i + n_per_row;
      break;
    case 1:
      c = (i % n_per_row) / A.meta().depth;
      if(c > 0) l = i - A.meta().depth;
      if(c < A.meta().width - 1) r = i + A.meta().depth;
      break;
    case 2:
      z = i % A.meta().depth;
      if(z > 0) l = i - 1;
      if(z < A.meta().depth - 1) r = i + 1;
    }
    if(l == i || r == i) cnt = 1;
    ptrB[i] = (ptrA[r] - ptrA[l]) / cnt;
  }

  return B;
}

vector<Image> ImageUtils::gradients(const Image A){
  IM_ARITH_B(A.type(), return, gradients(A));
}
template<typename T> vector<Image> _ImageUtils<T>::gradients(const Image A){
  int d = A.meta().dimensions;
  int n = A.meta().ncells;
  int* dlengths = new int[d];
  T** ptrs = new T*[d];
  
  vector<Image> grads;


#ifdef CUDA_GPU
  const Image *imgs[1] = {&A};
  mode m = Image::modeCheck(imgs, 1);
#endif

  

  vector<int> vdims = A.meta().dims();
  for(int i=0; i<d; i++){
    grads.push_back(Image(A.meta()));
#ifdef CUDA_GPU
    if(m == mode::gpu){
      ptrs[i] = (T*) grads.at(i).gptr();
    }else{
#endif
      ptrs[i] = grads.at(i).ptr<T>();
#ifdef CUDA_GPU
    }
#endif
    if(i == 0){
      dlengths[d-1] = 1;
    }else{
      dlengths[d-1-i] = dlengths[d-i] * vdims.at(d-i);  
    }
  }

#ifdef CUDA_GPU
  if(m == mode::gpu){
    for(int di=0; di<d; di++){
      cuda_gradient(n, (float*) A.gptr(), (float*) ptrs[di], dlengths[di], vdims.at(di));
    }   
  }else{
#endif
  
  
  T *ptrA = A.ptr<T>();
#pragma omp parallel for
  for(int i=0; i<n; i++){
    for(int di=0; di<d; di++){
      int idi = (i / dlengths[di]) % vdims.at(di);
      T t = 0 < idi && idi < vdims[di]-2 ? 2 : 1;
      T l = idi > 0 ? ptrA[i - dlengths[di]] : ptrA[i];
      T r = idi < vdims[di]-2 ? ptrA[i + dlengths[di]] : ptrA[i];
      ptrs[di][i] = (r - l) / t;     
    }
  }

#ifdef CUDA_GPU
  }
#endif

  delete[] ptrs;
  delete[] dlengths;
  return grads;  
}


/**
 * ===================================
 * Jacobian determinants
 * ===================================
 */
Image ImageUtils::jacobianDeterminants(const vector<Image> f){
  IM_ARITH_B(f.at(0).type(), return, jacobianDeterminants(f));
}
template<typename T> Image _ImageUtils<T>::jacobianDeterminants(const std::vector<Image> f){
  ImageMeta meta = f.at(0).meta();
  int dxd = meta.dimensions * meta.dimensions;
  T** ptrs = new T*[dxd];    
  
  std::vector<Image> grads; // [f1dx, f1dy, f1dz, f2dx, f2dy, f2dz, f3dx, f3dy, f3dz]


  for(uint fi=0; fi<meta.dimensions; fi++){
    std::vector<Image> fgrads = _ImageUtils<T>::gradients(f.at(fi));
    for(uint di=0; di<meta.dimensions; di++){
      Image grad = fgrads.at(di);
      grads.push_back(grad);
#ifdef CUDA_GPU
      if(meta.m == gpu){
	ptrs[fi * meta.dimensions + di] = (T*) grad.gptr();
      }else{ 
#endif
	ptrs[fi * meta.dimensions + di] = grad.ptr<T>();
#ifdef CUDA_GPU
      }
#endif
    }
  }
  
  Image determinants(meta);
  

#ifdef CUDA_GPU
  if(meta.m == gpu){
    float *gptr = determinants.gptr();
    switch(meta.dimensions){
    case 2:
      cuda_2x2_determinant(meta.ncells, (float**) ptrs, gptr);
      break;
    case 3:
      cuda_3x3_determinant(meta.ncells, (float**) ptrs, gptr);
      break;
    }
  }else{
#endif
  
  T* ptr = determinants.ptr<T>();

  #pragma omp parallel for
  for(uint i=0; i<meta.ncells; i++){
    switch(meta.dimensions){
    case 2:
      ptr[i] = ptrs[0][i] * ptrs[3][i] - ptrs[1][i] * ptrs[2][i];
      break;
    case 3:
      ptr[i] = ptrs[0][i]*ptrs[4][i]*ptrs[8][i]
        + ptrs[1][i]*ptrs[5][i]*ptrs[6][i]
        + ptrs[2][i]*ptrs[3][i]*ptrs[7][i]
        - ptrs[2][i]*ptrs[4][i]*ptrs[6][i]
        - ptrs[1][i]*ptrs[3][i]*ptrs[8][i]
        - ptrs[0][i]*ptrs[5][i]*ptrs[7][i];
    }
  }

#ifdef CUDA_GPU
  }
#endif
  
  delete[] ptrs;
  return determinants;
}

/**
 * ===================================
 * Pad
 * ===================================
 */
template<typename T> Image _ImageUtils<T>::pad(const Image A, int bwidth, PadType type){
  ImageMeta meta = A.meta();
  meta.width = meta.width + 2 * bwidth;
  meta.height = meta.height + 2 * bwidth;
  bool dim3 = meta.dimensions == 3;
  if(dim3){
    meta.depth = meta.depth + 2 * bwidth;
  }

  Image B = Image(meta);

  int nperrow_B = B.meta().width * B.meta().depth;
  int nperrow_A = A.meta().width * A.meta().depth;

  T *ptrA = A.ptr<T>();
  T *ptrB = B.ptr<T>();
  #pragma omp parallel for
  for(int r=0; r<(int) B.height(); r++){    
    for(int c=0; c<(int) B.width(); c++){
      for(int z=0; z<(int) B.depth(); z++){
        int ib = r * nperrow_B + c * B.depth() + z;
        switch(type){
        case zero:
          if(r >= bwidth && r< (int) A.height()+ bwidth && c >= bwidth && c < (int) A.width() + bwidth &&
             (!dim3 || (z >= bwidth && z<(int) A.depth() + bwidth))){
            int ra = r - bwidth;
            int ca = c - bwidth;
            int za = dim3 ? z - bwidth : 0;
            int ia = ra * nperrow_A + ca * A.depth() + za;	
            ptrB[ib] = ptrA[ia];
          }else{
            ptrB[ib] = (T) 0;
          }
          break;
        case mirror:
          int ra = min(max(r - bwidth, 0), (int) A.height()-1);
          int ca = min(max(c - bwidth, 0), (int) A.width()-1);
          int za = dim3 ? min(max(z - bwidth, 0), (int) A.depth()-1) : 0;
          int ia = ra * nperrow_A + ca * A.depth() + za;	
          ptrB[ib] = ptrA[ia];
        }	
      }
    }
  }  
  return B;
}
Image ImageUtils::pad(const Image A, int width, PadType type){
  IM_ARITH_B(A.type(), return, pad(A, width, type));
}



template<typename T> Image _ImageUtils<T>::unpad(const Image A, int bwidth){
  ImageMeta meta = A.meta();
  meta.width = meta.width - 2 * bwidth;
  meta.height = meta.height - 2 * bwidth;
  bool dim3 = meta.dimensions == 3;
  if(dim3){
    meta.depth = meta.depth - 2 * bwidth;
  }

  Image B = Image(meta);

  int nperrow_B = B.meta().width * B.meta().depth;
  int nperrow_A = A.meta().width * A.meta().depth;

  T *ptrA = A.ptr<T>();
  T *ptrB = B.ptr<T>();
  #pragma omp parallel for
  for(uint r=0; r<B.height(); r++){    
    for(uint c=0; c<B.width(); c++){
      for(uint z=0; z<B.depth(); z++){
	
        int ib = r * nperrow_B + c * B.depth() + z;
        int ia = (r+bwidth) * nperrow_A + (c+bwidth) * A.depth() + (dim3 ? z + bwidth : 0);
        ptrB[ib] = ptrA[ia];
		
      }
    }
  }  
  return B;
}
Image ImageUtils::unpad(const Image A, int width){
  IM_ARITH_B(A.type(), return, unpad(A, width));
}


/**
 * ===================================
 * Stringify
 * ===================================
 */
std::string ImageUtils::toString(const Image A){
  IM_ARITH_B(A.type(), return, toString(A));
}
template<typename T> std::string _ImageUtils<T>::toString(const Image A){
  T *ptrA = A.ptr<T>();
  std::ostringstream s;
  for(uint r=0; r<A.height(); r++){
    s << "[";
    for(uint c=0; c<A.width(); c++){
      if(A.depth() > 1) s << "[";
      for(uint z=0; z<A.depth(); z++){
        s << *ptrA;
        if(z != A.depth() - 1) s << " ";
        ptrA++;
      }
      if(A.depth() > 1) s << "]";
      else if(c != A.width() - 1) s << " ";
    }
    s << "]\n";
  }
  return s.str();
}

/**
 * ===================================
 * Reductions
 * ===================================
 */
template<typename T> double _ImageUtils<T>::sum(const Image A){
#ifdef CUDA_GPU
  if(A.meta().m == gpu){
    return (T) cuda_sum(A.meta().ncells, A.gptr());
  }
#endif
  double s = 0;
  T* p = A.ptr<T>();
#pragma omp parallel for reduction (+:s)
  for(uint i=0; i<A.meta().ncells; i++){
    s += (double) p[i];
  }
  return s;
}

double ImageUtils::sum(const Image A){
  IM_ARITH_B(A.type(), return,  sum(A));
}


/**
 * ===================================
 * FFT
 * ===================================
 */

Image ImageUtils::fft(const Image A){
  IM_ARITH_B(A.type(), return, fft(A));
}
template<typename T> Image _ImageUtils<T>::fft(const Image A){
  ImageMeta meta = A.meta();
  uint dims[3] = {meta.height, meta.width, meta.depth};

  ImageMeta metab = meta;
  metab.dtype = cpx;
  Image B = Image(metab);
  
#ifdef CUDA_GPU
  if(A.meta().m == gpu){
    cuda_fft(dims, A.gptr(), (cufftComplex*) B.gptr());
    return B;
  }  
#endif

  NOT_IMPLEMENTED;
}

Image ImageUtils::ifft(const Image A, ImageDataType dtype){
  IM_ARITH_B(dtype, return, ifft(A));
}

template<typename T> Image _ImageUtils<T>::ifft(const Image A){
  ImageMeta meta = A.meta();
  uint n = meta.ncells; 
  uint dims[3] = {meta.height, meta.width, meta.depth};
    
  ImageMeta metab = meta;
  metab.dtype = _ImageUtils<T>::dtype();
  Image B = Image(metab);

#ifdef CUDA_GPU
  if(A.meta().m == gpu){
    cuda_ifft(dims, (cufftComplex*) A.gptr(), (float*) B.gptr());
    ImageUtils::scalar(B, (float) 1/n, B); // Scale result
    return B;
  }  
#endif
  NOT_IMPLEMENTED;
}

/**
 * ===================================
 * FFT shift ops
 * ===================================
 */


Image ImageUtils::fftShift(const Image A){
  IM_ARITH_B(A.type(), return, fftShift(A));
}
template<typename T> Image _ImageUtils<T>::fftShift(const Image A){
  
  ImageMeta meta = A.meta();
  uint dims[3] = {meta.height, meta.width, meta.depth};

  Image B = Image(meta);

#ifdef CUDA_GPU
  if(A.meta().m == gpu){
    cuda_fft_shift(dims, (cufftComplex*) A.gptr(), (cufftComplex*) B.gptr());
    return B;
  }
#endif
  throw NotImplementedException();
}


template<> ImageDataType _ImageUtils<double>::dtype(){
  return f64;
}
template<> ImageDataType _ImageUtils<uint8_t>::dtype(){
  return uint8;
}
template<> ImageDataType _ImageUtils<complex>::dtype(){
  return cpx;
}
template<> ImageDataType _ImageUtils<float>::dtype(){
  return f32;
}




void ImageUtils::showImage(const Image A, int wait){
  IM_ARITH_A(A.type(), showImage(A, wait));
}


/**
 * ===================================
 *    ALPHA -> BETA IMAGE
 * ===================================
 */

void ImageUtils::betaImage(const Image A, Image alpha, Image B, float beta){

  #ifdef CUDA_GPU
  cuda_beta_image(A.meta().ncells, A.gptr(), alpha.gptr(), B.gptr(), beta);
  return;
  #endif
  NOT_IMPLEMENTED;
  
}


/**
 * ===================================
 *     Image operator overloads
 * ===================================
 */


Image operator*(complex s, const Image &A){ return A * s; }
Image operator*(double s, const Image &A){ return A * s; }
Image operator*(const Image &A, double s){
  Image target(A.meta());
  ImageUtils::scalar<double>(A, s, target);
  return target;
}
Image operator*(const Image &A, float s){
  Image target(A.meta());
  ImageUtils::scalar<float>(A, s, target);
  return target;
}
Image operator*(const Image &A, complex s){
  Image target(A.meta());
  ImageUtils::scalar<complex>(A, s, target);
  return target;
}

Image operator+(double s, const Image &A){ return A + s; }
Image operator+(const Image &A, double s){
  Image target(A.meta());
  ImageUtils::s_add<double>(A, s, target);
  return target;
}

Image operator+(const Image &A, float s){
  Image target(A.meta());
  ImageUtils::s_add<float>(A, s, target);
  return target;
}

Image operator+(complex s, const Image &A){ return A + s; }
Image operator+(const Image &A, complex s){
  Image target(A.meta());
  ImageUtils::s_add<complex>(A, s, target);
  return target;
}

Image operator+(const Image &A, const Image &B){
  Image target(A.meta());;
  ImageUtils::m_add(A, B, target);
  return target;
}

Image operator*(const Image &A, const Image &B){
  Image target(A.meta());
  ImageUtils::multiply(A, B, target);
  return target;
}


Image operator-(const Image &A, const Image &B){
  Image target(A.meta());
  ImageUtils::subtract(A, B, target);
  return target;
}

std::ostream& operator<< ( std::ostream& outs, const Image &A ){
  return outs << ImageUtils::toString(A);
}



/**
 * =======================================
 *     complex float operator overloads
 * =======================================
 */

template<> complex _ImageUtils<double>::toComplex(double a){
  return {a, 0};
}
template<> complex _ImageUtils<uint8_t>::toComplex(uint8_t a){
  return {(double) a, 0};
}
template<> complex _ImageUtils<complex>::toComplex(complex a){
  return a;
}
template<> complex _ImageUtils<float>::toComplex(float a){
  return {(double) a, 0};
}

template<> double _ImageUtils<double>::fromComplex(complex a){
  return a.r;
}
template<> uint8_t _ImageUtils<uint8_t>::fromComplex(complex a){
  return (int) a.r;
}
template<> complex _ImageUtils<complex>::fromComplex(complex a){
  return a;
}
template<> float _ImageUtils<float>::fromComplex(complex a){
  return a.r;
}

complex operator*(complex a, complex b){
  return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
complex operator*(int a, complex b){
  return {b.r * a, b.i * a};
}
complex operator-(complex a, complex b){
  return {a.r-b.r, a.i-b.i};
}
complex operator/(complex a, int s){
  return a / (double) s;
}
complex operator/(complex a, double s){
  return {a.r / s, a.i / s};
}
complex operator/(complex a, complex b){
  double denom = b.r * b.r + b.i * b.i;
  return {(a.r * b.r + a.i * b.i) / denom, (a.i*b.r - a.r*b.i)/denom};
}
complex operator*(complex a, double s){
  return {a.r * s, a.i * s};
}
complex operator+(complex a, complex b){
  return {a.r + b.r, a.i + b.i};
}
bool operator>(complex a, complex b){
  (void) a; (void) b;
  throw runtime_error("Undefined behaviour: complex ordering");
}
bool operator>(complex a, int b){
  return a.r > b;
}
ostream& operator<< ( std::ostream& outs, complex a){
  return outs << "(" << a.r << " + i" << a.i << ")";
}

ostream& operator<< ( std::ostream& outs, const ImageMeta &a){
  return outs << a.to_string();
}
ostream& operator<< ( std::ostream& outs, const Cube cube ) {
  return outs << "(" << cube.r0 << ", " << cube.c0 << ", " << cube.z0 << " ; " << \
    cube.r1 << ", " << cube.c1 << ", " << cube.z1 << ")";
}



/**
 * =======================================
 *    Basic utils
 * =======================================
 */

std::chrono::_V2::system_clock::time_point Utils::_tic;
std::chrono::_V2::system_clock::time_point  Utils::tic(){
  Utils::_tic = Clock::now();
  return _tic;
}

std::chrono::microseconds Utils::toc(std::chrono::_V2::system_clock::time_point  t1){
  auto t2 = tic();
  auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
  std::cout << dt.count() << "us\n";
  return dt;
}

std::chrono::microseconds Utils::toc(){
  auto t1 = Utils::_tic;
  auto t2 = tic();
  auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
  std::cout << dt.count() << "us\n";
  return dt;
}

#ifdef WITH_OPENCV
#include <OpenCVInt.hpp>

template<typename T> void _ImageUtils<T>::showImage(const Image A, int wait){
  Image A2 = A;
#ifdef CUDA_GPU
  A2.toCPU();
#endif
  if(A2.dimensions()==2){
    OpenCVInt<T>::showImage(A2, wait);
  }else{
    for(uint i=0; i<A2.depth(); i++){
      Image slice = A2.slice<T>(0, A2.height(), 0, A2.width(), i, i+1);
      OpenCVInt<T>::showImage(slice, wait);
    }
  }
}

void Utils::showImage(const Image A, int wait){
  ImageUtils::showImage(A, wait);
}
#else
void Utils::showImage(const Image A, int wait){
  std::cout << "Show image undefined...\n";
}

#endif

