#ifndef MSE_H
#define MSE_H

/**
 * \file mse.h GPU implementation for MSE measure.
 **/

/**
 * Computes the MSE alpha measure.
 * \param n Number of elements.
 * \param eta The eta regularization value.
 * \param g_Sinv The inverse target image.
 * \param g_M The moving image.
 * \param g_alpha The alpha channel.
 **/
float cuda_mse_alpha_sum(int n, float eta, float *g_Sinv, float *g_M, float *g_alpha);

#endif
