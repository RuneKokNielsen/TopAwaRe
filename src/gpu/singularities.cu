#include "gpu.h"
#include "singularities.h"

/**
 * ==================================================================
 * Singularity operations
 * ==================================================================
 **/
/**
 * Point singularity Walpha
 **/

#define fill_alpha()				\
  alpha[i] = max(0.0, min(alpha[i], ((di - pi + abs/2)/(abs))));

#define fill_Walpha(a)				\
  float factor;							\
  if(di/pi <= 1){						\
    factor = 1;							\
  }else{							\
    float r = (di/pi - 1) / sigma;				\
    factor = r >= 1 ? 0 : pi * (1 - r) * (1 - r) / di;		\
  }								\
								\
  for(int j=0; j<a; j++){					\
    XW[j][i] -= factor * ds[j];					\
  }								\

#define fill_WalphaInv(a)						\
  float factor;								\
  if(di/pi >= sigma + 1 || di == 0){						\
    factor = 0;								\
  }else{								\
    float r = (di/pi);							\
    factor = pi * ((-bq + sqrt(bq*bq - 4 * aq * (cq - r)))/(2*aq) - r) / di; \
  }									\
  									\
  for(int j=0; j<a; j++){						\
    XW[j][i] += factor * ds[j];						\
  }									\

#define pointSingularity_dist(a)				\
  float ds[a];							\
  float di = 0;							\
  for(int j=0; j<a; j++){					\
    ds[j] = X[j][i] - x[j];					\
    di += ds[j] * ds[j];					\
  }								\
  di = sqrt(di);						

#define kernel_pointSingularity_fillWalpha(a)	\
  int i = blockIdx.x*blockDim.x + threadIdx.x;	\
  if (i < n){							\
								\
    pointSingularity_dist(a);					\
								\
    fill_Walpha(a);						\
  }								\


  
__global__
void kernel_pointSingularity_fillWalpha_3d(int n, mesh X, mesh XW, x3 x, float pi, float sigma){
  kernel_pointSingularity_fillWalpha(3);
}
__global__
void kernel_pointSingularity_fillWalpha_2d(int n, mesh X, mesh XW, x3 x, float pi, float sigma){
  kernel_pointSingularity_fillWalpha(2);
}
void cuda_pointSingularity_fillWalpha(int n, int dims, float** XW, float ** X, float *Xp, float pi, float sigma){
  switch(dims){
  case 2:
    kernel_pointSingularity_fillWalpha_2d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), mesh(XW), x3(Xp), pi, sigma);
    break;
  case 3:
    kernel_pointSingularity_fillWalpha_3d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), mesh(XW), x3(Xp), pi, sigma);
  };
  cuda_error_handle(); 
}


#define kernel_pointSingularity_fillWalphaInv(a)			\
  int i = blockIdx.x*blockDim.x + threadIdx.x;				\
  if (i < n){								\
									\
    pointSingularity_dist(a);						\
    fill_WalphaInv(a);							\
									\
									\
  }									\

__global__
void kernel_pointSingularity_fillWalphaInv_3d(int n, mesh X, mesh XW, x3 x, float pi, float sigma, float aq, float bq, float cq){
  kernel_pointSingularity_fillWalphaInv(3);
}
__global__
void kernel_pointSingularity_fillWalphaInv_2d(int n, mesh X, mesh XW, x3 x, float pi, float sigma, float aq, float bq, float cq){
  kernel_pointSingularity_fillWalphaInv(2);
}
void cuda_pointSingularity_fillWalphaInv(int n, int dims, float** XW, float ** X, float *Xp, float pi, float sigma){
  float a = -1.0/(sigma*sigma);
  float b = 1 + 2.0/sigma + 2.0/(sigma*sigma);
  float c = -1.0/(sigma*sigma) - 2.0/sigma - 1;
  switch(dims){
  case 2:
    kernel_pointSingularity_fillWalphaInv_2d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), mesh(XW), x3(Xp), pi, sigma, a, b, c);
    break;
  case 3:
    kernel_pointSingularity_fillWalphaInv_3d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), mesh(XW), x3(Xp), pi, sigma, a, b, c);
  };
  cuda_error_handle(); 
}



/**
 * Point singularity alpha
 **/
#define kernel_pointSingularity_fillAlpha(a)	\
  int i = blockIdx.x*blockDim.x + threadIdx.x;	\
  if (i < n){								\
    pointSingularity_dist(a);						\
    fill_alpha();							\
  }

__global__
void kernel_pointSingularity_fillAlpha_3d(int n, mesh X, float *alpha, x3 x, float pi, float abs){
  kernel_pointSingularity_fillAlpha(3);
}
__global__
void kernel_pointSingularity_fillAlpha_2d(int n, mesh X, float *alpha, x3 x, float pi, float abs){
  kernel_pointSingularity_fillAlpha(2);
}
void cuda_pointSingularity_fillAlpha(int n, int dims, float ** X, float *alpha, float *Xp, float pi, float abs){
  switch(dims){
  case 2:
    kernel_pointSingularity_fillAlpha_2d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), alpha, x3(Xp), pi, abs);
    break;
  case 3:
    kernel_pointSingularity_fillAlpha_3d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), alpha, x3(Xp), pi, abs);
  }
}





__device__
float lineseg_dist(int dims, float lambda, x3 x0, x3 x1, x3 x, x3& ds, float& di, bool is_plane, x3 plane_normal){
  x3 x01;
  float squared_norm01 = 0;
  float t = 0;
  for(int i=0; i<dims; i++){
    x01[i] = x1[i] - x0[i];
    squared_norm01 += x01[i] * x01[i];
    ds[i] = x[i] - x0[i];
    t += ds[i] * x01[i];
  }
  t = t / squared_norm01;
  t = max(0.0, min(1.0, t));

  x3 xp;
  di = 0;
  for(int i=0; i<dims; i++){
    xp[i] = x0[i] + t * x01[i];
    ds[i] = x[i] - xp[i];
    di += ds[i] * ds[i];
  }
  di = sqrt(di);

  // If lambda !=0 and t in {0, 1}, compute the dot product
  // of unit vectors to weight the distance
  if(lambda > 0 && (t == 0 || t == 1) && di > 0){
    float dot = 0;
    if(is_plane){
      for(int i=0; i<dims; i++){
	dot += ds[i]/di * plane_normal[i];
      }
    }else{
      for(int i=0; i<dims; i++){
	dot += ds[i]/di * x01[i]/sqrt(squared_norm01);
      }
    }
    
    di = di * (1 + lambda * pow(dot, 2));
  }

  return 1;
}

__device__
float lineseg_dist(int dims, float lambda, x3 x0, x3 x1, x3 x, x3& ds, float& di){
  return lineseg_dist(dims, lambda, x0, x1, x, ds, di, false, x3());
}

#define lineSingularity_dist(a)			\
  x3 ds;					\
  float di;					\
  x3 x(X, i);					\
  pi *= lineseg_dist(a, lambda, x0, x1, x, ds, di);

#define kernel_lineSingularity_fillAlpha(a)	\
  int i = blockIdx.x*blockDim.x + threadIdx.x;	\
  if (i < n){					\
    						\
    lineSingularity_dist(a);						\
									\
    fill_alpha();							\
  }
__global__
void kernel_lineSingularity_fillAlpha_2d(int n, mesh X, float *alpha, x3 x0, x3 x1, float pi, float abs, float lambda){
  kernel_lineSingularity_fillAlpha(2);
}

__global__
void kernel_lineSingularity_fillAlpha_3d(int n, mesh X, float *alpha, x3 x0, x3 x1, float pi, float abs, float lambda){
  kernel_lineSingularity_fillAlpha(3);
}

void cuda_lineSingularity_fillAlpha(int n, int dims, float ** X, float *alpha, float *Xp, float pi, float abs, float lambda){
  switch(dims){
  case 2:
    kernel_lineSingularity_fillAlpha_2d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), alpha, x3(Xp), x3(Xp, 3), pi, abs, lambda);
    break;
  case 3:
    kernel_lineSingularity_fillAlpha_3d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), alpha, x3(Xp), x3(Xp, 3), pi, abs, lambda);
    break;
  }
}

#define kernel_lineSingularity_fillWalpha(a)				\
  int i = blockIdx.x*blockDim.x + threadIdx.x;				\
  if (i < n){								\
    lineSingularity_dist(a);						\
    fill_Walpha(a);							\
  }

__global__
void kernel_lineSingularity_fillWalpha_2d(int n, mesh X, mesh XW, x3 x0, x3 x1, float pi, float sigma, float lambda){
  kernel_lineSingularity_fillWalpha(2);
}

__global__
void kernel_lineSingularity_fillWalpha_3d(int n, mesh X, mesh XW, x3 x0, x3 x1, float pi, float sigma, float lambda){
  kernel_lineSingularity_fillWalpha(3);
}

void cuda_lineSingularity_fillWalpha(int n, int dims, float** XW, float ** X, float *Xp, float pi, float sigma, float lambda){
  switch(dims){
  case 2:
    kernel_lineSingularity_fillWalpha_2d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), mesh(XW), x3(Xp), x3(Xp+3), pi, sigma, lambda);
    
    break;
  case 3:
    kernel_lineSingularity_fillWalpha_3d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), mesh(XW), x3(Xp), x3(Xp+3), pi, sigma, lambda);
    break;
  };
  cuda_error_handle(); 
}

#define kernel_lineSingularity_fillWalphaInv(a)				\
  int i = blockIdx.x*blockDim.x + threadIdx.x;				\
  if (i < n){								\
    lineSingularity_dist(a);						\
    fill_WalphaInv(a);							\
  }


__global__
void kernel_lineSingularity_fillWalphaInv_2d(int n, mesh X, mesh XW, x3 x0, x3 x1, float pi, float sigma, float aq, float bq, float cq, float lambda){
  kernel_lineSingularity_fillWalphaInv(2);
}
__global__
void kernel_lineSingularity_fillWalphaInv_3d(int n, mesh X, mesh XW, x3 x0, x3 x1, float pi, float sigma, float aq, float bq, float cq, float lambda){
  kernel_lineSingularity_fillWalphaInv(3);
}
void cuda_lineSingularity_fillWalphaInv(int n, int dims, float** XW, float ** X, float *Xp, float pi, float sigma, float lambda){
  float a = -1.0/(sigma*sigma);
  float b = 1 + 2.0/sigma + 2.0/(sigma*sigma);
  float c = -1.0/(sigma*sigma) - 2.0/sigma - 1;
  switch(dims){
  case 2:
    kernel_lineSingularity_fillWalphaInv_2d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), mesh(XW), x3(Xp), x3(Xp, 3), pi, sigma, a, b, c, lambda);
    break;
  case 3:
    kernel_lineSingularity_fillWalphaInv_3d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), mesh(XW), x3(Xp), x3(Xp, 3), pi, sigma, a, b, c, lambda);
    break;
  };
  cuda_error_handle(); 
}


#define planeSingularity_dist()						\
  float y[3] = {0};							\
  x3 x(X, i);								\
  x3 x0x = x - x0;							\
  for(int j=0; j<3; j++){						\
    for(int k=0; k<3; k++){						\
      y[j] += Binv[j][k] * x0x[k];					\
    }									\
  }									\
  x3 v1 = x1 - x0;							\
  x3 v2 = x2 - x0;							\
									\
  float di;								\
  x3 ds;								\
  if(0<=y[0] && y[0]<=1 && 0<=y[1] && y[1]<=1){				\
    float xp[3];							\
    for(int j=0; j<3; j++){						\
      xp[j] = x0[j] + y[0] * v1[j] + y[1] * v2[j];			\
    }									\
									\
    di = y[2] < 0 ? -y[2] : y[2];					\
    for(int j=0; j<3; j++){						\
      ds[j] = X[j][i] - xp[j];						\
    }									\
    di = y[2] < 0 ? -y[2] : y[2];					\
  }else{								\
    x3 pts[4];								\
    pts[0] = x0;							\
    pts[1] = x1;							\
    pts[2] = x0 + v1 + v2;						\
    pts[3] = x2;							\
    x3 x(X, i);								\
    float di_i;								\
    x3 ds_i;								\
    x3 normal;								\
    for(int j=0; j<3; j++){						\
      normal[j] = B[j][2];						\
    }									\
    for(int j=0; j<4; j++){						\
      lineseg_dist(3, lambda, pts[j], pts[(j+1)%4], x, ds_i, di_i, true, normal); \
      if(j == 0 || di_i < di){						\
	di = di_i;							\
	ds = ds_i;							\
      }									\
    }									\
  }									\



__global__
void kernel_planeSingularity_fillAlpha_3d(int n, mesh X, float *alpha, x3 x0, x3 x1, x3 x2, float pi, float abs, x33 Binv, x33 B, float lambda){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n){

    planeSingularity_dist();
     
    fill_alpha();
  }
}
void cuda_planeSingularity_fillAlpha(int n, int dims, float ** X, float *alpha, float *Xp, float pi, float abs, const float Binv[3][3], const float B[3][3], float lambda){
  switch(dims){
  case 2:    
    break;
  case 3:
    kernel_planeSingularity_fillAlpha_3d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), alpha, x3(Xp), x3(Xp, 3), x3(Xp, 6), pi, abs, x33(Binv), x33(B), lambda);
    break;
  }
  cuda_error_handle(); 
}

__global__
void kernel_planeSingularity_fillWalpha_3d(int n, mesh X, mesh XW, x3 x0, x3 x1, x3 x2, float pi, float sigma, x33 Binv, x33 B, float lambda){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n){
    planeSingularity_dist();

    fill_Walpha(3);

  }
}
void cuda_planeSingularity_fillWalpha(int n, int dims, float** XW, float ** X, float *Xp, float pi, float sigma, const float Binv[3][3], const float B[3][3], float lambda){
  switch(dims){
  case 2:
    break;
  case 3:
    kernel_planeSingularity_fillWalpha_3d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), mesh(XW), x3(Xp), x3(Xp, 3), x3(Xp, 6), pi, sigma, x33(Binv), x33(B), lambda);
    break;
  };
  cuda_error_handle(); 
}

__global__
void kernel_planeSingularity_fillWalphaInv_3d(int n, mesh X, mesh XW, x3 x0, x3 x1, x3 x2, float pi, float sigma, float aq, float bq, float cq, x33 Binv, x33 B, float lambda){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n){
    
    planeSingularity_dist();
    fill_WalphaInv(3);
    

  }
}
void cuda_planeSingularity_fillWalphaInv(int n, int dims, float** XW, float ** X, float *Xp, float pi, float sigma, const float Binv[3][3], const float B[3][3], float lambda){
  float a = -1.0/(sigma*sigma);
  float b = 1 + 2.0/sigma + 2.0/(sigma*sigma);
  float c = -1.0/(sigma*sigma) - 2.0/sigma - 1;
  switch(dims){
  case 2:    
    break;
  case 3:
    kernel_planeSingularity_fillWalphaInv_3d<<<nblocks(n),blocksize()>>>
      (n, mesh(X), mesh(XW), x3(Xp), x3(Xp, 3), x3(Xp, 6), pi, sigma, a, b, c, x33(Binv), x33(B), lambda);
    break;
  };
  cuda_error_handle(); 
}

