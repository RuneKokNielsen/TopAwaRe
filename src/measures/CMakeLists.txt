
add_library(metrics SSD.cpp L2alpha.cpp ControlPointSD.cpp ControlPointAD.cpp )
target_link_libraries( metrics base )
