#include <LinearInterpolator.hpp>
#include <math.h>

template<typename T> void _LinearInterpolator<T>::interpolate(const Image in, const vector<Image> mesh, Image out){
  ImageMeta meta = in.meta();
  if(in.dimensions() == 2){

    double rmax = ((1.0 / meta.height) * (meta.height - 1)) * meta.height;
    double cmax = ((1.0 / meta.width) * (meta.width - 1)) * meta.width;

#ifdef CUDA_GPU
    if(meta.m == gpu){
      cuda_2d_linear_interpolation(mesh.at(0).meta().ncells, in.gptr(), out.gptr(), mesh.at(0).gptr(), mesh.at(1).gptr(), meta.height, meta.width);
      return;
    }
#endif

    T *pin = in.ptr<T>();
    T *pout = out.ptr<T>();
    T *dr = mesh.at(0).ptr<T>();
    T *dc = mesh.at(1).ptr<T>();

   

#pragma omp parallel for
    for(unsigned int i=0; i<mesh.at(0).meta().ncells; i++){


      // Scale coordinates up to N-by-M
      double rlm = dr[i] *  meta.height;
      rlm = min(rmax, max(0.0, rlm));
      //      rlm = min((double) meta.height - 3, max(0.0, rlm));
      double clm = dc[i] * meta.width;
      clm = min(cmax, max(0.0, clm));
      //clm = min((double) meta.width, max(0.0, clm));

      // Get surrounding pixel coordinates
      unsigned int rl = floor(rlm);
      unsigned int cl = floor(clm);   
      
      // Get pixel coordinates within boundary
      unsigned int ra = min(rl, meta.height-1);
      unsigned int ca = min(cl, meta.width-1);
      unsigned int rd = min(meta.height-1, ra + 1);
      unsigned int cd = min(meta.width-1, ca + 1);
      
      
      double wr = rlm - rl;
      double wc = clm - cl;

      pout[i] =
	pin[ra * meta.width + ca] * (1-wr) * (1-wc)
	+ pin[ra * meta.width + cd] * (1-wr) * wc
	+ pin[rd * meta.width + ca] * wr * (1-wc)
	+ pin[rd * meta.width + cd] * wr * wc;

      
    }    

  }else if(in.dimensions() == 3){





    int nperrow = meta.width * meta.depth;

    double rmax = ((1.0 / meta.height) * (meta.height - 1)) * meta.height;
    double cmax = ((1.0 / meta.width) * (meta.width - 1)) * meta.width;
    double zmax = ((1.0 / meta.depth) * (meta.depth - 1)) * meta.depth;
    rmax = meta.height - 1;
    cmax = meta.width - 1;
    zmax = meta.depth - 1;


#ifdef CUDA_GPU
    if(meta.m == gpu){
      cuda_3d_linear_interpolation(mesh.at(0).meta().ncells, in.gptr(), out.gptr(), mesh.at(0).gptr(), mesh.at(1).gptr(), mesh.at(2).gptr(), meta.height, meta.width, meta.depth);
      return;
    }
#endif
    

    T *pin = in.ptr<T>();
    T *pout = out.ptr<T>();
    T *dr = mesh.at(0).ptr<T>();
    T *dc = mesh.at(1).ptr<T>();
    T *dz = mesh.at(2).ptr<T>();
    
#pragma omp parallel for
    for(unsigned int i=0; i<meta.ncells; i++){


      // Scale coordinates up to N-by-M
      double rlm = dr[i] * meta.height;
      rlm = min(rmax, max(0.0, rlm));
      double clm = dc[i] * meta.width;
      clm = min(cmax, max(0.0, clm));
      double zlm = dz[i] * meta.depth;
      zlm = min(zmax, max(0.0, zlm));      

      // Get surrounding pixel coordinates
      unsigned int rl = floor(rlm);
      unsigned int cl = floor(clm);
      unsigned int zl = floor(zlm);

      
      // Get pixel coordinates within boundary
      int ra = min(rl, meta.height - 2);
      int ca = min(cl, meta.width - 2);
      int za = min(zl, meta.depth - 2);      
      
      double wr = rlm - rl;
      double wc = clm - cl;
      double wz = zlm - zl;
      
      pout[i] =
	  pin[ra * nperrow + ca * meta.depth + za] * (1-wr)*(1-wc)*(1-wz)
	+ pin[ra * nperrow + ca * meta.depth + za + 1] * (1-wr)*(1-wc)*wz
	+ pin[ra * nperrow + (ca + 1) * meta.depth + za] * (1-wr)*wc*(1-wz)
	+ pin[ra * nperrow + (ca + 1) * meta.depth + za + 1] * (1-wr)*wc*wz
	+ pin[(ra+1) * nperrow + ca * meta.depth + za] * wr*(1-wc)*(1-wz)
	+ pin[(ra+1) * nperrow + ca*meta.depth + za + 1] * wr*(1-wc)*wz
	+ pin[(ra+1) * nperrow + (ca+1)*meta.depth + za] * wr*wc*(1-wz)
	+ pin[(ra+1) * nperrow + (ca+1)*meta.depth + za + 1] *wr*wc*wz;      
    }
    
  }
#pragma omp flush
}
