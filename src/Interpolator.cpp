#include <Interpolator.hpp>
#include <math.h>


template<typename T> void _Interpolator<T>::interpolate(InterpolationMethod m, const Image in, const vector<Image> mesh, Image out){


  ImageMeta meta = in.meta();
  if(in.dimensions() == 2){

    T *pin = in.ptr<T>();
    T *pout = out.ptr<T>();
    double *dr = mesh.at(0).ptr<double>();
    double *dc = mesh.at(1).ptr<double>();
    double rx, cx;

    #pragma omp parallel for
    for(int i=0; i<meta.ncells; i++){


      // Scale coordinates up to N-by-M
      double rlm = dr[i] *  meta.height;
      double clm = dc[i] * meta.width;

      // Get surrounding pixel coordinates
      int rl = floor(rlm);
      int rr = rl + 1;
      int cl = floor(clm);
      int cr = clm + 1;      
      
      // Get pixel coordinates within boundary
      int ra = min(max(rl, 0), meta.height - 1);
      int ca = min(max(cl, 0), meta.width - 1);
      int rd = min(max(rl + 1, 0), meta.height - 1);
      int cd = min(max(cl + 1, 0), meta.width - 1);
      
      
      double wr = rlm - rl;
      double wc = clm - cl;
      
      pout[i] =
	pin[ra * meta.width + ca] * (1-wr) * (1-wc)
	+ pin[ra * meta.width + cd] * (1-wr) * wc
	+ pin[rd * meta.width + ca] * wr * (1-wc)
	+ pin[rd * meta.width + cd] * wr * wc;
      

    }

    
  }else if(in.dimensions() == 3){

  }

}


