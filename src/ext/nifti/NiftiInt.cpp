
#include <NiftiInt.hpp>


using namespace NiftiInt;

template<typename T1> Image Nifti<T1>::readImage(string path){  

  
  // Read the image metadata (but not pixel data) from file


  FSLIO *fslio;
  void *buffer;

  fslio = FslOpen(path.c_str(),"rb");

  char* cpath = new char[path.size() + 1];
  path.copy(cpath, path.size());
  cpath[path.size()] = '\0';
  cout << cpath << endl;
  cout << path << endl;
  buffer = FslReadAllVolumes(fslio, cpath);
  delete[] cpath;


  short x, y, z, v;
  FslGetDim(fslio, &x, &y, &z, &v);
  int nvols = x * y * z;;
  cout << "volume: " << nvols << "\n";
  cout << x << " " << y << " " << z << " " << v << "\n";
  short t;
  FslGetDataType(fslio, &t);


  ImageMeta meta;

  size_t dimensionality;
  FslGetDimensionality(fslio, &dimensionality);
  
  meta.height = x;
  meta.width = y;
  meta.depth = z;
  meta.dimensions = dimensionality;
  meta.dtype = typeToImgType<T1>();;

  Image im(meta, false);
  
  cout << t << "\n";
  // See datatypes in niftilib/nifti1.h
  switch(t){
  case 64: // double
    Nifti<T1>().transfer<double>((double*) buffer, im);
    break;
  case 16: // float
    Nifti<T1>().transfer<float>((float*) buffer, im);
    break;
  case 4: // signed short
    Nifti<T1>().transfer<short>((short*) buffer, im);
    break;
  case 2: // uint8
    Nifti<T1>().transfer<uint8_t>((uint8_t*) buffer, im);
    break;
  }        
  FslClose(fslio);
  
  return im; 
}


template<typename T1> void Nifti<T1>::writeImage(Image im, string path_src, string path_target){

  FSLIO *fslio_src;
  fslio_src = FslOpen(path_src.c_str(), "rb");
  FSLIO *fslio_target;
  fslio_target = FslOpen(path_target.c_str(), "wb");

  // Copy header information from source file
  FslCloneHeader(fslio_target, fslio_src);

  ImageMeta meta = im.meta();
  FslSetDim(fslio_target, meta.height, meta.width, meta.depth, 1);

  FslSetDataType(fslio_target, 16);
  
  short t;
  FslGetDataType(fslio_target, &t);
  char *buffer;
  buffer = new char[meta.ncells * t];
  switch(t){
  case 64: // double    
    Nifti<T1>().transfer<double>(im, (double*) buffer);
    break;
  case 16: // float
    Nifti<T1>().transfer<float>(im, (float*) buffer);
    break;
  case 4: // short
    Nifti<T1>().transfer<short>(im, (short*) buffer);
    break;
  case 2: // uint8
    Nifti<T1>().transfer<uint8_t>(im, (uint8_t*) buffer);
    break;
  }          
  FslWriteAllVolumes(fslio_target,buffer);
  delete buffer;
  
  FslClose(fslio_src);
  FslClose(fslio_target);
}


template<typename T1>
template<typename T2>
void Nifti<T1>::transfer(T2 *data, Image target){
  T1 *p = target.ptr<T1>();

  int h = target.meta().height;
  int w = target.meta().width;
  int d = target.meta().depth;

  int ppca = d;
  int ppra = w * ppca;
  
  int ppcb = h;
  int ppzb = ppcb * w;


#pragma omp parallel for
  for(unsigned int i=0; i<target.meta().ncells; i++){
    int x = i % ppcb;
    int y = floor((i % ppzb) / ppcb);
    int z = floor(i / ppzb);

    int j = x * ppra + y * ppca + z;
    p[j] = (T1) data[i];
  }

}

template<typename T1>
template<typename T2>
void Nifti<T1>::transfer(Image im, T2 *buffer){

  T1 *p = im.ptr<T1>();

  int h = im.meta().height;
  int w = im.meta().width;
  int d = im.meta().depth;

  int ppca = d;
  int ppra = w * ppca;
  
  int ppcb = h;
  int ppzb = ppcb * w;


#pragma omp parallel for
  for(unsigned int i=0; i<im.meta().ncells; i++){
    int x = i % ppcb;
    int y = floor((i % ppzb) / ppcb);
    int z = floor(i / ppzb);
    
    int j = x * ppra + y * ppca + z;
    buffer[i] = Nifti<T1>().convert<T2>(p[j]);
  }
}

template<typename T1>
template<typename T2>
T2 Nifti<T1>::convert(T1 a){
  return (T2) a;
}

