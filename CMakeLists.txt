#===========================================
# COMPILATION OPTIONS
#==========================================


# Use GPU in parallelization?
set(with_gpu 1)
# see https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
add_definitions(-DCUDA_BLOCKSIZE=1024)

# Build with opencv?
set(with_opencv 1)
# Build with Nifti reader?
set(with_nifti 1)

# Build with boost stacktrace?
set(with_stacktrace 0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


#==========================================



cmake_minimum_required(VERSION 2.8)
project( TopAwaReProject )

find_package( OpenCV )

set (CMAKE_CXX_STANDARD 11)

# Core includes
include_directories(inc inc/transformation_models inc/kernels inc/gradients inc/interpolators inc/integration_schemes inc/measures inc/utils inc/configuration)
subdirs(src)


# Build with OpenMP parallelization enabled
set (CMAKE_CXX_FLAGS "-fopenmp")

# Build warnings
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wnon-virtual-dtor -pedantic")

# Build with debug compability
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

if(with_nifti)
  include_directories(inc/ext/nifti)
  include_directories(ext/niftilib/include)
endif()
if(with_opencv)
  include_directories(inc/ext/opencv)
endif()


if(with_gpu)
  add_definitions(-DCUDA_GPU)
  find_package(CUDA QUIET REQUIRED)

  cuda_add_executable( topaware JSONMain.cpp )
  set (CUDA_PROPAGATE_HOST_FLAGS OFF)
  set_property(TARGET topaware PROPERTY CUDA_STANDARD 11)

  include_directories(inc/gpu)
  cuda_add_library( libgpu "src/gpu/gpu.cu" "src/gpu/singularities.cu" "src/gpu/fft.cu" "src/gpu/cc.cu" "src/gpu/image.cu" "src/gpu/mse.cu" OPTIONS -arch sm_50 -Xcompiler -fPIC)
  CUDA_ADD_CUFFT_TO_TARGET( libgpu )

  target_link_libraries( topaware libgpu )
elseif(1)
  add_executable( topaware JSONMain.cpp )
endif()



target_link_libraries( topaware src )

# json library
include_directories(ext/nlohmann inc/ext/json)
subdirs(src/ext/json)
target_link_libraries( topaware ext_json )


#Build with boost program_options
target_link_libraries( topaware "-lboost_program_options -ldl" )


install(TARGETS topaware DESTINATION bin)




#=========================================
#         Optional dependencies
#========================================






#----------------------------------------
if(with_opencv)
  add_definitions(-DWITH_OPENCV)
  include_directories(inc/ext/opencv)
  subdirs(src/ext/opencv)
  target_link_libraries( topaware ext_opencv )
endif()

if(with_nifti)
  add_definitions(-DHAVE_ZLIB)
  add_definitions(-DWITH_NIFTI)
  include_directories(inc/ext/nifti)

  find_library(Z_LIB z)
 
  

  # Build library
  execute_process(COMMAND mkdir -p ext/niftilib)
  execute_process(COMMAND wget https://sourceforge.net/projects/niftilib/files/latest/download -O ext/nifticlib-2.0.0.tar.gz)
  execute_process(COMMAND tar -xvzf nifticlib-2.0.0.tar.gz -C niftilib --strip-components 1 WORKING_DIRECTORY ext)
  execute_process(COMMAND sh ext_fpic_fix.sh)
  execute_process(COMMAND make WORKING_DIRECTORY ext/niftilib)
  include_directories(ext/niftilib/include)


  execute_process(COMMAND mkdir -p ext/nlohmann)
  execute_process(COMMAND wget https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp -O ext/nlohmann/json.hpp)
  
  find_library(FSLIO_LIB fslio ext/niftilib/lib)
  find_library(NIFTIIO_LIB niftiio ext/niftilib/lib)
  find_library(ZNZ_LIB znz ext/niftilib/znzlib)

  
  subdirs(src/ext/nifti)
  target_link_libraries(topaware ext_nifti)
  
endif()

if(with_stacktrace)
  add_definitions(-DWITH_STACKTRACE)
endif()


