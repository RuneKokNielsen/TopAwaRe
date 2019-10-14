#ifndef JSON_CONFIGURATION_TREE_HPP
#define JSON_CONFIGURATION_TREE_HPP

#include "ConfigurationTree.hpp"
#include "json.hpp"
using json = nlohmann::json;

/**
 * JSON backend for interfacing with JSON configuration files.
 * Based on the nlohmann JSON implementation: https://github.com/nlohmann/json
 **/
class JSONConfigurationTree: ConfigurationTreeBackend{

public:
  JSONConfigurationTree(){};
  JSONConfigurationTree(json j): _j(j){};
  ~JSONConfigurationTree(){};
  template<typename T> T get(string field);

  string getString(string field);
  float getFloat(string field);
  vector<float> getVecFloat(string field);
  vector<vector<float>> getVecVecFloat(string field);
  vector<vector<vector<float>>> getVecVecVecFloat(string field);
  int getInt(string field);
  vector<int> getVecInt(string field);
  bool getBool(string field);

  JSONConfigurationTree* getChild(string field);
  vector<ConfigurationTreeBackend*> getChildren(string field);

  bool hasField(string field);

  JSONConfigurationTree* loadFile(string path);
  void merge(ConfigurationTreeBackend* conf);

protected:

  json _j;

};

template<typename T>
T JSONConfigurationTree::get(string field){
  try{
    return _j[field].get<T>();
  }catch(nlohmann::detail::exception e){
    throw ConfigurationException("Error reading key " + field + " : " +  e.what());
  }
}



#endif
