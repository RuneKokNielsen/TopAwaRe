
add_library( config Node.cpp ConfigurationTree.cpp )
target_link_libraries( config metrics )
