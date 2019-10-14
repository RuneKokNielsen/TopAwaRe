#ifndef GPU_IMAGE_H
#define GPU_IMAGE_H

/**
 * \file image.h GPU implementations of common image functions.
 **/


/**
 * Copy the real part of a complex array.
 * \param n Number of elements.
 * \param a The complex array to copy from.
 * \param b The float array to copy to.
 */
void cuda_complex_real(int n, cufftComplex* a, float* b);

/**
 * Copy float array to the real part of a complex array,
 * and setting the imaginary part to 0.
 * \param n Number of elements.
 * \param a Float array to copy from.
 * \param b Complex array to copy to.
 **/
void cuda_real_complex(int n, float* a, cufftComplex* b);


/**
 * Element-wise addition of float arrays.
 * \param n Number of elements.
 * \param g_a One array.
 * \param g_b Another array.
 * \param g_c Target array.
 **/
void cuda_madd(int n, float* g_a, float* g_b, float* g_c);
/**
 * Element-wise subtraction of float arrays.
 * \param n Number of elements.
 * \param g_a One array.
 * \param g_b Another array.
 * \param g_c Target array.
 **/
void cuda_msub(int n, float* g_a, float* g_b, float* g_c);
/**
 * Element-wise multiplication of float arrays.
 * \param n Number of elements.
 * \param g_a One array.
 * \param g_b Another array.
 * \param g_c Target array.
 **/
void cuda_mult(int n, float* g_a, float* g_b, float* g_c);
/**
 * Element-wise multiplication of complex arrays.
 * \param n Number of elements.
 * \param g_a One array.
 * \param g_b Another array.
 * \param g_c Target array.
 **/
void cuda_mult_cpx(int n, cufftComplex* g_a, cufftComplex* g_b, cufftComplex* g_c);
/**
 * Element-wise division of float arrays.
 * \param n Number of elements.
 * \param g_a One array.
 * \param g_b Another array.
 * \param g_c Target array.
 **/
void cuda_div(int n, float* g_a, float* g_b, float* g_c);
/**
 * Element-wise division of complex arrays.
 * \param n Number of elements.
 * \param g_a One array.
 * \param g_b Another array.
 * \param g_c Target array.
 **/
void cuda_div_cpx(int n, cufftComplex* g_a, cufftComplex* g_b, cufftComplex* g_c);
/**
 * Scaling of float array.
 * \param n Number of elements.
 * \param g_a Input array.
 * \param s Scalar value.
 * \param g_b Target array.
 **/
void cuda_scalar(int n, float* g_a, float s, float* g_b);
/**
 * Scalar addition of scalar to float array.
 * \param n Number of elements.
 * \param g_a Input array.
 * \param s Scalar value to add.
 * \param g_b Target array.
 **/
void cuda_sadd(int n, float* g_a, float s, float* g_b);
/**
 * Element-wise max between each element and a scalar value.
 * \param n Number of elements.
 * \param g_a Input array.
 * \param g_b Target array.
 * \param min The new minimum value of each element.
 **/
void cuda_max(int n, float* g_a, float* g_b, float min);


/**
 * Finite difference gradient of image in some direction.
 * \param n Number of elements.
 * \param g_a Input image.
 * \param g_b Output gradient image.
 * \param dlength Spacing in array between adjacent "rows" in this direction.
 * \param vdim Number of voxels in this dimension.
 **/
void cuda_gradient(int n, float* g_a, float* g_b, int dlength, int vdim);
/**
 * Compute matrix determinant at each element in array of 2x2
 * matrices.
 * \param n Number of elements.
 * \param g_as Pointer to array of 4 float arrays corresponding to the 4 elements
 * of each 2x2 matrix.
 * \param g_b Target array of determinants.
 **/
void cuda_2x2_determinant(int n, float** g_as, float* g_b);

/**
 * Compute matrix determinant at each element in array of 3x3
 * matrices.
 * \param n Number of elements.
 * \param g_as Pointer to array of 9 float arrays corresponding to the 9 elements
 * of each 3x3 matrix.
 * \param g_b Target array of determinants.
 **/
void cuda_3x3_determinant(int n, float** g_as, float* g_b);

/**
 * Fill a cartesian meshgrid in [0, 1] intervals
 * \param n Number of elements.
 * \param g_as Array of target images.
 * \param d Number of dimensions (2 or 3)
 * \param dims Array of dimensionalities (measured in voxels).
 * \param deltas Array of voxel spacings.
 **/
void cuda_meshgrid(int n, float** g_as, uint d, uint* dims, float* deltas);

/**
 * Copy the values from one array to another.
 * \param n Number of elements.
 * \param a Source array.
 * \param b Target array.
 **/
void cuda_copy(int n, float* a, float* b);

/**
 * Copy the values of a slice from one image to another image.
 * \param n Number of elements in target image.
 * \param from Source data.
 * \param to Target data.
 * \param h0 Source height.
 * \param w0 Source width.
 * \param d0 Source depth.
 * \param h1 Target height.
 * \param w1 Target width.
 * \param d2 Target depth.
 * \param r0 First row of slice in source image.
 * \param c0 First column of slice in source image.
 * \param z0 First depth coordinate of slice in source image.
 **/
void cuda_copy_slice(int n, float* from, float* to, int h0, int w0, int d0, int h1, int w1, int d1, int r0, int c0, int z0);

/**
 * Linear interpolation of 3D data.
 * \param n Number of elements in interpolation mesh.
 * \param g_in Source image to be interpolated from.
 * \param g_out Result of interpolation.
 * \param g_dr Row mesh.
 * \param g_dc Column mesh.
 * \param g_dz Depth mesh.
 * \param dimr Height of source image.
 * \param dimc Width of source image.
 * \param dimz Depth of source image.
 **/
void cuda_3d_linear_interpolation(int n, float* g_in, float* g_out, float* g_dr, float* g_dc, float* g_dz, int dimr, int dimc, int dimz);
/**
 * Linear interpolation of 2D data.
 * \param n Number of elements in interpolation mesh.
 * \param g_in Source image to be interpolated from.
 * \param g_out Result of interpolation.
 * \param g_dr Row mesh.
 * \param g_dc Column mesh.
 * \param dimr Height of source image.
 * \param dimc Width of source image.
 **/
void cuda_2d_linear_interpolation(int n, float* g_in, float* g_out, float* g_dr, float* g_dc, int dimr, int dimc);

/**
 * Return the maximal magnitude of any vector in
 * a vector field.
 * \param n Number of elements.
 * \param ndims Number of dimensions (2 or 3)
 * \param v0 Vector dimension 0.
 * \param v1 Vector dimension 1.
 * \param v2 Vector dimension 2 (if ndims == 3)
 **/
float cuda_maximal_magnitude(int n, int ndims, float* v0, float* v1, float* v2);

/**
 * Summation of float data
 * \param n Number of elements.
 * \param g_a Data to sum over.
 **/
float cuda_sum(int n, float* g_a);

/**
 * Given an image, an alpha channel and a background
 * value beta, returns the image with transparency corresponding
 * to the alpha channel and beta as background intensity.
 * \param n Number of elements.
 * \param a Source image.
 * \param alpha The alpha channel.
 * \param b Target image.
 * \param beta The background beta value.
 **/
void cuda_beta_image(int n, float* a, float* alpha, float* b, float beta);



#endif
