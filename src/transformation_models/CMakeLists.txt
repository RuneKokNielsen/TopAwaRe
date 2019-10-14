
add_library(optimizers LDDMMOptimizer.cpp SingularityOptimizer.cpp Singularities.cpp ) #CompositionOptimizer.cpp SingularityOptimizer.cpp)
target_link_libraries(optimizers src)
