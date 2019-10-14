#ifndef REGISTRATION_FRAME_HPP
#define REGISTRATION_FRAME_HPP

#include "Node.hpp"
#include "Registration.hpp"

/**
 * Root node in the registration execution tree.
 **/
class RegistrationFrame: Conf::Node{

public:

  /**
   * Executes the configured registration execution tree on
   * the given data. Handles reading and preparing data,
   * executing the registration using the registration sub-
   * tree and writing the results to disk.
   * \param sourcePath Path to source image.
   * \param targetPath Path to target image.
   * \param outPath Path to output directory.
   **/
  void execute(string sourcePath, string targetPath, string outPath);


  /**
   * Loads an image from file.
   * \param path Path to file.
   **/
  Image loadImage(string path);

  /**
   * Writes an image to file.
   * \param im The image to write.
   * \param sourcePath The path of a file containing a similar image file. Only relevant for nifti files.
   * \param outPath Path of output file.
   **/
  void writeImage(Image im, string sourcePath, string outPath);

  RegistrationFrame(){};
  void load(ConfigurationTree conf);

private:


  void writeResult(tuple<Image, Image, vector<Image>, vector<Image>, vector<Image>, vector<Image>, measure::Context* > result, std::chrono::microseconds t);

  Registration* _registration;

  string _pathI;
  string _pathS;
  string _pathOut;

  bool _normalize;
  vector<int> _slice;
  int _padding;


};

#endif
