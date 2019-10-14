

add_library(ext_nifti NiftiInt.cpp)
target_link_libraries(ext_nifti ${FSLIO_LIB} ${ZNZ_LIB} ${Z_LIB} ${NIFTIIO_LIB})
