#ifndef GPU_H
#define GPU_H

/**
 * \file gpu.h
 * GPU implementation essentials.
 **/


#include <string>
#include <cufft.h>
#include <stdio.h>

#include "singularities.h"
#include "fft.h"
#include "cc.h"
#include "mse.h"
#include "image.h"

/**
 * Exception signifying an error in GPU execution.
 **/
struct CudaException : public std::exception
{
  std::string _str;
 CudaException(std::string str) : _str(str) {}
  ~CudaException() throw () {} 
  const char* what() const throw() { return _str.c_str(); }
};

/**
 * Device wrapper for 3D coordinate mesh.
 **/
struct mesh{
  mesh(float **_X){
    for(int i=0; i<3; i++){
      X[i] = _X[i];
    }
  }

  __device__
  float*& operator[](const int i){
    return X[i];
  }

  float *X[3];
};

/**
 * Device wrapper of 3D index.
 **/
struct i3{

  i3(int *_x){
    for(int i=0; i<3; i++){
      x[i] = _x[i];
    }
  }

  __device__
  int operator[](const int i){
    return x[i];
  }

  int x[3];
};

/**
 * Device wrapper of 3D coordinate.
 **/
struct x3{

  __device__
  x3(){};

  /**
   * Set coordinate as the offset'th
   * value in 3D mesh.
   **/
  x3(float *_x, int offset=0){
    for(int i=0; i<3; i++){
      x[i] = _x[i + offset];
    }
  };

  __device__
  x3(mesh X, int i){
    for(int j=0; j<3; j++){
      x[j] = X[j][i];
    }
  }

  __device__
  x3 operator+(x3 b){
    x3 c;
    for(int i=0; i<3; i++){
      c[i] = x[i] + b[i];
    }
    return c;
  };

  __device__
  x3 operator-(const x3 b){
    x3 c;
    for(int i=0; i<3; i++){
      c[i] = x[i] - b[i];
    }
    return c;
  }

  __device__
  float& operator[](const int i){
    return x[i];
  }

  __device__
  float operator[](const int i) const{
    return x[i];
  }

  float x[3];
};

/**
 * Device wrapper for 3x3 matrix.
 **/
struct x33{
  x33(const float _x[3][3]){
    for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
        x[i][j] = _x[i][j];
      }
    }
  }

  __device__
  float* operator[](const int i){
    return x[i];
  }

  float x[3][3];
};



/**
 * Allocate float array in GPU memory.
 * \param n Number of elements.
 * \param zerofill If true, initializes all elements to 0.
 **/
float *cuda_f32_allocate(int n, bool zerofill=false);
/**
 * Free an array in GPU memory.
 * \param g_a Pointer to start of array.
 **/
void cuda_f32_free(float* g_a);
/**
 * Transfer float data from CPU memory to GPU memory.
 * \param n Number of elements.
 * \param a Pointer to start of CPU array.
 * \param g_a Pointer to start of GPU array.
 **/
void cuda_f32_send(int n, float* a, float* g_a);
/**
 * Transfer float data from GPU memory to CPU memory.
 * \param n Number of elements.
 * \param a Pointer to start of CPU array.
 * \param g_a Pointer to start of GPU array.
 **/
void cuda_f32_retrieve(int n, float *a, float *g_a);


/**
 * The configured cuda device blocksize (see CMakeLists.txt)
 **/
int blocksize();
/**
 * Returns the number of required blocks for a given
 * number of elements.
 * \param n Number of elements to process.
 **/
int nblocks(int n);
/**
 * Handles any errors after interaction with cuda.
 **/
void cuda_error_handle();
/**
 * Sets the device to be used by the application.
 * \param devid The id of the device to use.
 **/
void cuda_set_device(int devid);



#endif
