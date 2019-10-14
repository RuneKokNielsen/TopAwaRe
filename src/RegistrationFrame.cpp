#include "RegistrationFrame.hpp"

#include "NiftiInt.hpp"
#include "OpenCVInt.hpp"
#include "System.hpp"
#include "Measure.hpp"

#include "Global.hpp"

#include <fstream>

#ifdef CUDA_GPU
#include "gpu.h"
#endif

void RegistrationFrame::load(ConfigurationTree conf){

  conf.requireFields({"registration"});

  _registration = (Registration*) Conf::loadTree(conf, "registration");
  _normalize = conf.get<bool>("normalize", false);
  _slice = conf.get<vector<int>>("slice", {});
  _padding = conf.get<int>("padding", 0);

#ifdef CUDA_GPU
  cuda_set_device(conf.get<int>("cuda_device", 0));
#endif
}

void RegistrationFrame::execute(string sourcePath, string targetPath, string outPath){

  _pathI = sourcePath;
  _pathS = targetPath;
  _pathOut = outPath;

  SOURCE_PATH = sourcePath;

  Image I = loadImage(_pathI);
  Image S = loadImage(_pathS);

#ifdef CUDA_GPU
  I.toGPU();
  S.toGPU();
#endif
  
  auto t0 = TIC;
  tuple<Image, Image, vector<Image>, vector<Image>, vector<Image>, vector<Image>, measure::Context* > result = _registration->apply(S, I);

  auto dt = tTOC(t0);
  writeResult(result, dt);
}



Image RegistrationFrame::loadImage(string path){
  Image im;
  if(path.find(".nii")!=std::string::npos){
    im = NiftiInt::Nifti<float>::readImage(path);
  }else{
    im = OpenCVInt<float>::readImage(path);
  }

  
  if(_normalize) ImageUtils::normalize(im);
  if(_slice.size()){
    if(_slice.size() != 6){
      throw invalid_argument("slice input must contain exactly 6 points: r0, r1, c0, c1, z0, z1");
    }
    Cube cube = Cube(_slice);
    im = ImageUtils::slice(im, cube);
  }
  if(_padding > 0){
    im = ImageUtils::pad(im, _padding, zero);
  }
  
  return im;
}

void RegistrationFrame::writeImage(Image im, string sourcePath, string outPath){
  if(sourcePath.find(".nii") != string::npos){
    NiftiInt::Nifti<float>::writeImage(im, sourcePath, outPath + ".nii.gz");
  }else{
    OpenCVInt<float>::writeImage(im, outPath + ".png");
  }
}

void RegistrationFrame::writeResult(tuple<Image, Image, vector<Image>, vector<Image>, vector<Image>, vector<Image>, measure::Context* > result, std::chrono::microseconds dt){
  
  System::prepareOutputDir(_pathOut);


  ofstream tfile;
  tfile.open(_pathOut + "/time");
  tfile << dt.count() << "\n";
  tfile.close();
  
  Image imres = get<0>(result);
  Image iminv = get<1>(result);
  vector<Image> forward = get<2>(result);
  vector<Image> backward = get<3>(result);
  vector<Image> diffeo = get<4>(result);
  vector<Image> invdiffeo = get<5>(result);
#ifdef CUDA_GPU
  imres.toCPU();
  iminv.toCPU();
#endif  
  imres = ImageUtils::unpad(imres, _padding);
  iminv = ImageUtils::unpad(iminv, _padding);

  
  if(!_pathOut.empty()){
    cout << "Writing result to " << _pathOut << "\n";

    // Write resulting images
    writeImage(imres, _pathI, _pathOut + "/M");
    writeImage(iminv, _pathI, _pathOut + "/Sinv");

    // Write transformations
    for(uint i=0; i<imres.meta().dimensions; i++){
#ifdef CUDA_GPU
      forward.at(i).toCPU();
      backward.at(i).toCPU();      
      diffeo.at(i).toCPU();
      invdiffeo.at(i).toCPU();     
#endif
      diffeo.at(i) = ImageUtils::unpad(diffeo.at(i), _padding);
      forward.at(i) = ImageUtils::unpad(forward.at(i), _padding);
      backward.at(i) = ImageUtils::unpad(backward.at(i), _padding);
      invdiffeo.at(i) = ImageUtils::unpad(invdiffeo.at(i), _padding);
            
      if(_pathI.find(".nii")!=string::npos){	
        NiftiInt::Nifti<float>::writeImage(forward.at(i), _pathI, _pathOut + "/W_" + to_string(i) + ".nii.gz");
        NiftiInt::Nifti<float>::writeImage(backward.at(i), _pathI, _pathOut + "/W_inv_" + to_string(i) + ".nii.gz");
      }else{
        OpenCVInt<float>::writeCsv(forward.at(i), _pathOut + "/W_" + to_string(i) + ".csv");
        OpenCVInt<float>::writeCsv(invdiffeo.at(i), _pathOut + "/Wdiffeo_inv_" + to_string(i) + ".csv");
      }
    }


    measure::AlphaContext* context = dynamic_cast<measure::AlphaContext*>(get<6>(result));
    if(context != NULL){
      // Write alpha map
      cout << "Writing alpha map to alpha_" << _pathOut << "\n";
      Image alpha = dynamic_cast<measure::AlphaContext*>(get<6>(result))->getContext(0)->alpha;
#ifdef CUDA_GPU
      alpha.toCPU();
#endif
      alpha = ImageUtils::unpad(alpha, _padding);
      if(_pathI.find(".nii")!=string::npos){
        Image alphainv = Image(alpha.meta(), true);
        ImageUtils::s_add(alphainv, 1.0, alphainv);
        ImageUtils::subtract(alphainv, alpha, alphainv);
        NiftiInt::Nifti<float>::writeImage(alphainv, _pathI, _pathOut + "/alpha.nii.gz");
      }else{
        OpenCVInt<float>::writeImage(alpha, _pathOut + "/alpha.png");
      }
    }

  }

  
}
