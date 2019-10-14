#ifndef GPU_CC_H
#define GPU_CC_H

/**
 * \file cc.h GPU implementations for cross correlation measure.
 **/


/**
 * Precompute helper variables for easy computation of gradient and measure.
 * \param n Number of voxels.
 * \param M Pointer to moving image.
 * \param Sinv Pointer to inverse target image.
 * \param A Pointer to target A construct.
 * \param B Pointer to target B construct.
 * \param C Pointer to target C construct.
 * \param Mbar Pointer to target mean normalized moving image.
 * \param Sinvbar Pointer to target mean normalized inverse target image.
 * \param dimx Image height.
 * \param dimy Image width.
 * \param dimz Image depth.
 * \param w Computation window radius.
 **/
void cuda_cc_preproc(int n, float* M, float* Sinv, float* A, float* B, float* C, float* Mbar, float* Sinvbar, int dimx, int dimy, int dimz, int w);

/**
 * Compute the gradient given precomputed variables.
 * \param n Number of voxels.
 * \param Mbar Pointer to mean normalized moving image.
 * \param Sinvbar Pointer to mean normalized inverse target image.
 * \param A Pointer to A construct.
 * \param B Pointer to B construct.
 * \param C Pointer to C construct.
 * \param target Pointer to target gradient placeholder.
 **/
void cuda_cc_grad(int n, float* Mbar, float* Sinvbar, float* A, float* B, float* C, float* target);

/**
 * Compute th measure given precomputed variables.
 * \param n Number of voxels.
 * \param A Pointer to A construct.
 * \param B Pointer to B construct.
 * \param C Pointer to C construct.
 * \param target Pointer to target local measure placeholder.
 **/
void cuda_cc_measure(int n, float* A, float* B, float* C, float* target);

#endif
