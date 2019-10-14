/**
 * \file JSONMain.cpp
 * Application entry point. Creates and runs an execution tree based
 * on the json configuration file supplied in the input arguments.
 **/


#include <signal.h>     // ::signal, ::raise
#include <iostream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "json.hpp"
using json = nlohmann::json;


#include "Node.hpp"
#include "JSONConfigurationTree.hpp"
#include "RegistrationFrame.hpp"

using namespace std;

#include <fstream>

#ifdef WITH_STACKTRACE
#include <boost/stacktrace.hpp>
#endif

/**
 * Global signal handler. If build WITH_STACKTRACE,
 * stacktraces are printed for unhandled exceptions.
 **/
void signal_handler(int signum) {
  ::signal(signum, SIG_DFL);
#ifdef WITH_STACKTRACE
  cerr << boost::stacktrace::stacktrace();
#endif
  ::raise(SIGABRT);
}

/**
 * Application entry point. Creates and runs an execution tree based
 * on the json configuration file supplied in the input arguments.
 **/
int main(int argc, char* argv[]){

  ::signal(SIGSEGV, &signal_handler);
  ::signal(SIGABRT, &signal_handler);

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "Print this message")
    ("source", po::value<string>()->required(), "Path to source image")
    ("target", po::value<string>()->required(), "Path to target image")
    ("out", po::value<string>()->required(), "Path to output")
    ("configuration", po::value<string>()->required(), "Path to configuration file");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if(vm.count("help")){
    cout << desc << "\n";
    exit(0);
  }
  try{
    po::notify(vm);
  }catch(po::required_option e){
    cout << "\nINPUT ERROR: " << e.what() << "\n\n" << desc << "\n";
    exit(1);
  }

  string confFile = vm["configuration"].as<string>();
  ConfigurationTree conf((ConfigurationTreeBackend*) JSONConfigurationTree().loadFile(confFile));
  RegistrationFrame* reg = (RegistrationFrame*) Conf::loadTree(conf, "root");
  reg->execute(vm["source"].as<string>(), vm["target"].as<string>(), vm["out"].as<string>());

  return 0;
}
