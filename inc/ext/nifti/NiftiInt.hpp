#ifndef NIFTI_INT_HPP
#define NIFTI_INT_HPP

#include <Image.hpp>

extern "C" {
#include <fslio.h>
}

namespace NiftiInt{

  /**
   * Basic nifti file I/O for 3D images.
   * Based on niftilib: http://niftilib.sourceforge.net/
   **/
  template<typename T1>
  class Nifti
  {


  public:
    /**
     * Read the nifti file into an Image object.
     * \param path Path to nifti file.
     **/
    static Image readImage(string path);

    /**
     * Write Image object data to nifti file.
     * \param im Image object to write.
     * \param path_src Path to existing nifti file. The nifti configuration will be copied from this file.
     * \param apth_target Path to new nifti file.
     **/
    static void writeImage(Image im, string path_src, string path_target);

  private:

    /**
     * Convert and transfer raw image data of type T2 from a nifti file to an image object
     * with data type T1.
     * \param buffer Image data from source nifti file.
     * \param target Image object to store data in.
     **/
    template<typename T2> void transfer(T2 *buffer, Image target);
    /**
     * Convert and transfer image data of type T1 from an image object to a buffer to be
     * saved to a nifti file.
     * \param src The image object to transfer data from.
     * \param buffer The target buffer to transfer data to.
     **/
    template<typename T2> void transfer(Image src, T2 *buffer);
    /**
     * Convert value of type T1 to type T2.
     * \param a Value to convert.
     **/
    template<typename T2> T2 convert(T1 a);
  };




  template class Nifti<float>;

}




#endif
