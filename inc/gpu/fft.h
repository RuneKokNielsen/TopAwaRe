#ifndef GPU_FFT_H
#define GPU_FFT_H

/**
 * FFT implementation based on cuFFT.
 **/

/**
 * FFT based on cuFFT.
 * \param dims Image dimensions.
 * \param g_in Pointer to input float image.
 * \param g_out Pointer to output complex (2 x float) image.
 **/
void cuda_fft(unsigned int* dims, float* g_in, cufftComplex* g_out);

/**
 * Inverse FFT based on cuFFT.
 * \param dims Image dimensions.
 * \param g_in Pointer to input complex image.
 * \param g_out Pointer to output float image.
 **/
void cuda_ifft(unsigned int* dims, cufftComplex* g_in, float* g_out);

/**
 * Zero-frequency shift of data in Fourier domain.
 * \param dims Image dimensions.
 * \param g_in Pointer to input complex image.
 * \param g_out Pointer to output complex image.
 **/
void cuda_fft_shift(unsigned int* dims, cufftComplex* g_in, cufftComplex* g_out);

#endif
