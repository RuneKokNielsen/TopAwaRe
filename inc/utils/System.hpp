#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <string>
#include <iostream>
#include <sys/stat.h>
using namespace std;
namespace System{


  inline void prepareOutputDir(string out){

    struct stat dirinfo;
    if( stat( out.c_str(), &dirinfo ) != 0 ){
      cout << "Create output directory '" << out << "'..\n";
      if(mkdir(out.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) < 0){
	cout << "Failed to make output directory!\n";
	exit(-1);
      }
    }   
    else if( dirinfo.st_mode & S_IFDIR ){
      // Directory exists, do nothing
    }
    else{
      cout << "Cannot access directory..\n";
      exit(-1);
    }
  }
  
}


#endif
