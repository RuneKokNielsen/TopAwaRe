
#include <OpenCVInt.hpp>
#include <fstream>

template<typename T> Image OpenCVInt<T>::toImage(Mat m){

  ImageMeta meta = ImageMeta(
    (uint) m.rows,
    (uint) m.cols,
    (uint) m.channels(),
    (uint) (m.channels() > 1 ? 3 : 2),
    Imaging::typeToImgType<T>(),
    sizeof(T)
			     );
  Image im(meta);

  T *p1 = m.ptr<T>(0);
  T *p2 = im.ptr<T>();
  for(uint i=0; i<meta.ncells; i++){
    *p2 = *p1;
    p1++;
    p2++;
  }
  return im;
}

template<> Image OpenCVInt<double>::readImage(std::string path){
  Mat a = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
  a.convertTo(a, CV_64F, 1.0/255.5);
  return toImage(a);
}
template<> Image OpenCVInt<float>::readImage(std::string path){
  Mat a = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
  a.convertTo(a, CV_32F, 1.0/255.5);
  return toImage(a);
}


template<typename T> Mat OpenCVInt<T>::toMat(Image im, bool normalize){
  Mat m(im.height(), im.width(), CV_64F);
  T *p1 = im.ptr<T>();
  double *p2 = m.ptr<double>(0);
  for(uint i=0; i<im.meta().ncells; i++){
    p2[i] = (double) p1[i];
  }


  if(normalize){
    // If not in [0,1] spectrum the image is normalized  
    double min;
    double max;
    minMaxIdx(m, &min, &max);
    m -= min;  
    minMaxIdx(m, &min, &max);
    m = (m / max);
  
  }
  
  return m;
}

template<typename T> void OpenCVInt<T>::showImage(Image im, int wait){

  Mat m = toMat(im);
 
  
  resize(m, m, Size(600,600));
  
  namedWindow("Image", WINDOW_AUTOSIZE);
  imshow("Image", m);
  waitKey(wait);
}



template<typename T> void OpenCVInt<T>::writeImage(Image im, string path){
  Mat m = OpenCVInt<T>::toMat(im, true) * 255;
  imwrite(path, m);
}

template<typename T> void OpenCVInt<T>::writeCsv(Image im, string path){
  Mat m = OpenCVInt<T>::toMat(im, false);
  writeMatToFile(m, path.c_str());
}

template<typename T>
void OpenCVInt<T>::writeMatToFile(cv::Mat& m, const char* filename)
{
    ofstream fout(filename);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(uint i=0; i<(uint) m.rows; i++)
      {
        for(uint j=0; j<(uint) m.cols; j++)
          {
            fout<<m.at<double>(i,j);
            if(j<(uint)m.cols-1) fout << ",";
          }
        fout<<endl;
      }

    fout.close();
}
