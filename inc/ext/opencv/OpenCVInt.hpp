#ifndef OPEN_CV_INT_HPP
#define OPEN_CV_INT_HPP


#include <Image.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

/**
 * Implementation of 2D image I/O based on opencv2.
 **/
template <typename T>
class OpenCVInt{


public:

  /**
   * Read image from file into image object.
   * \param path Path to file.
   **/
  static Image readImage(string path);

  /**
   * Write image object to file.
   * \param im Image object to write.
   * \param path Path to new file.
   **/
  static void writeImage(Image im, string path);

  /**
   * Write image data to csv file. This is useful
   * for saving 2D displacement data.
   * \param im Image object containing data to write.
   * \param path Path to new csv file.
   **/
  static void writeCsv(Image im, string path);

  /**
   * Display the given image object and wait some time
   * before continuing.
   * \param im Image to be shown.
   * \param wait milliseconds to wait after displaying a frame.
   **/
  static void showImage(Image im, int wait);

private:

  /**
   * Convert opencv Mat object to image object.
   * \param m Mat object to convert.
   **/
  static Image toImage(Mat m);

  /**
   * Convert image object to opencv Mat object.
   * \param im Image to convert.
   * \param normalize If true, normalize image data to [0 1].
   */
  static Mat toMat(Image im, bool normalize=true);

  /**
   * Write opencv Mat object to file.
   * \param m Mat object to write.
   * \param filename Path to new file.
   **/
  static void writeMatToFile(cv::Mat& m, const char* filename);
};

template class OpenCVInt<float>;
template class OpenCVInt<double>;
template class OpenCVInt<uint8_t>;
template class OpenCVInt<Imaging::complex>;

#endif
