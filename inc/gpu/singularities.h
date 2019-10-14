#ifndef GPU_SINGULARITIES_H
#define GPU_SINGULARITIES_H


/**
 * \file singularities.h GPU implementation of singularity operations
 **/

/**
 * Compute the displacement caused by a point singularity.
 * \param n Number of elements.
 * \param dims Number of dimensions (2 or 3).
 * \param XW Current transformation mesh - to be updated.
 * \param X Current transformation mesh - unchanged.
 * \param Xp Coordinates of point.
 * \param pi Hole radius.
 * \param sigma Displacement smoothness.
 **/
void cuda_pointSingularity_fillWalpha(int n, int dims, float** XW, float** X, float* Xp,float pi, float sigma);
/**
 * Compute the inverse displacement caused by a point singularity.
 * \param n Number of elements.
 * \param dims Number of dimensions (2 or 3).
 * \param XW Current transformation mesh - to be updated.
 * \param X Current transformation mesh - unchanged.
 * \param Xp Coordinates of point.
 * \param pi Hole radius.
 * \param sigma Displacement smoothness.
 **/
void cuda_pointSingularity_fillWalphaInv(int n, int dims, float** XW, float** X, float* Xp,float pi, float sigma);
/**
 * Compute the alpha changes caused by a point singularity.
 * \param n Number of elements.
 * \param dims Number of dimensions (2 or 3).
 * \param X Current transformation mesh - unchanged.
 * \param Xp Coordinates of point.
 * \param pi Hole radius.
 * \param abs Alpha boundary smoothness factor.
 **/
void cuda_pointSingularity_fillAlpha(int n, int dims, float ** X, float *alpha, float *Xp, float pi, float abs);

/**
 * Compute the displacement caused by a line singularity.
 * \param n Number of elements.
 * \param dims Number of dimensions (2 or 3).
 * \param XW Current transformation mesh - to be updated.
 * \param X Current transformation mesh - unchanged.
 * \param Xp Coordinates of endpoints.
 * \param pi Hole radius.
 * \param sigma Displacement smoothness.
 * \param lambda Endpoint shape control.
 **/
void cuda_lineSingularity_fillWalpha(int n, int dims, float** XW, float** X, float* Xp,float pi, float sigma, float lambda);
/**
 * Compute the inverse displacement caused by a line singularity.
 * \param n Number of elements.
 * \param dims Number of dimensions (2 or 3).
 * \param XW Current transformation mesh - to be updated.
 * \param X Current transformation mesh - unchanged.
 * \param Xp Coordinates of endpoints.
 * \param pi Hole radius.
 * \param sigma Displacement smoothness.
 * \param lambda Endpoint shape control.
 **/
void cuda_lineSingularity_fillWalphaInv(int n, int dims, float** XW, float** X, float* Xp,float pi, float sigma, float lambda);
/**
 * Compute the alpha changes caused by a line singularity
 * \param n Number of elements.
 * \param dims Number of dimensions (2 or 3).
 * \param X Current transformation mesh - unchanged.
 * \param Xp Coordinates of endpoints.
 * \param pi Hole radius.
 * \param abs Alpha boundary smoothness.
 * \param lambda Endpoint shape control.
 **/
void cuda_lineSingularity_fillAlpha(int n, int dims, float** X, float* alpha, float* Xp, float pi, float abs, float lambda);

/**
 * Compute the displacement caused by a plane singularity.
 * \param n Number of elements.
 * \param dims Number of dimensions (2 or 3).
 * \param XW Current transformation mesh - to be updated.
 * \param X Current transformation mesh - unchanged.
 * \param Xp Coordinates of endpoints.
 * \param pi Hole radius.
 * \param sigma Displacement smoothness.
 * \param Binv Inverse of matrix B used for projection points onto the quadrilateral plane.
 * \param B Matrix B = [p0p1, p0p2, normal/|normal|] describing the quadrilateral.
 * \param lambda Endpoint shape control.
 **/
void cuda_planeSingularity_fillWalpha(int n, int dims, float** XW, float** X, float* Xp, float pi, float sigma, const float Binv[3][3], const float B[3][3], float lambda);
/**
 * Compute the inverse displacement caused by a plane singularity.
 * \param n Number of elements.
 * \param dims Number of dimensions (2 or 3).
 * \param XW Current transformation mesh - to be updated.
 * \param X Current transformation mesh - unchanged.
 * \param Xp Coordinates of endpoints.
 * \param pi Hole radius.
 * \param sigma Displacement smoothness.
 * \param Binv Inverse of matrix B used for projection points onto the quadrilateral plane.
 * \param B Matrix B = [p0p1, p0p2, normal/|normal|] describing the quadrilateral.
 * \param lambda Endpoint shape control.
 **/
void cuda_planeSingularity_fillWalphaInv(int n, int dims, float** XW, float** X, float* Xp, float pi, float sigma, const float Binv[3][3], const float B[3][3],float lambda);

/**
 * Compute the alpha changes caused by a plane singularity.
 * \param n Number of elements.
 * \param dims Number of dimensions (2 or 3).
 * \param X Current transformation mesh - unchanged.
 * \param Xp Coordinates of endpoints.
 * \param pi Hole radius.
 * \param abs Alpha boundary smoothness.
 * \param Binv Inverse of matrix B used for projection points onto the quadrilateral plane.
 * \param B Matrix B = [p0p1, p0p2, normal/|normal|] describing the quadrilateral.
 * \param lambda Endpoint shape control.
 **/
void cuda_planeSingularity_fillAlpha(int n, int dims, float** X, float* alpha, float* Xp, float pi, float abs, const float Binv[3][3], const float B[3][3], float lambda);





#endif
