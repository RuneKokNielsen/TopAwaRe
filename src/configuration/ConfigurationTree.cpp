#include "ConfigurationTree.hpp"
#include <stdexcept>


ConfigurationTree ConfigurationTree::getChild(string field){
  return ConfigurationTree(_backend->getChild(field));
}

vector<ConfigurationTree> ConfigurationTree::getChildren(string field){
  vector<ConfigurationTree> res;
  vector<ConfigurationTreeBackend*> vec = _backend->getChildren(field);
  for(unsigned int i=0; i<vec.size(); i++){
    res.push_back(ConfigurationTree(vec.at(i)));
  }
  return res;
}

bool ConfigurationTree::hasField(string field){
  return _backend->hasField(field);
}




template<> string ConfigurationTree::get<string>(string field){
  return _backend->getString(field);
}
template<> float ConfigurationTree::get<float>(string field){
  return _backend->getFloat(field);
}
template<> vector<float> ConfigurationTree::get<vector<float> >(string field){
  return _backend->getVecFloat(field);
}
template<> vector<vector<float>> ConfigurationTree::get<vector<vector<float> > >(string field){
  return _backend->getVecVecFloat(field);
}
template<> vector<vector<vector<float>>> ConfigurationTree::get<vector<vector<vector<float>>>>(string field){
  return _backend->getVecVecVecFloat(field);
}
template<> int ConfigurationTree::get<int>(string field){
  return _backend->getInt(field);
}
template<> vector<int> ConfigurationTree::get<vector<int> >(string field){
  return _backend->getVecInt(field);
}
template<> bool ConfigurationTree::get<bool>(string field){
  return _backend->getBool(field);
}

bool ConfigurationTree::getFlag(string field){
  return hasField(field) && get<bool>(field);
}


void ConfigurationTree::requireFields(vector<string> fields){

  for(unsigned int i=0; i<fields.size(); i++){

    if(!_backend->hasField(fields.at(i))){
      throw ConfigurationException("Missing field: " + fields.at(i));
    }
  }
  
}

ConfigurationTree ConfigurationTree::loadFile(string path){
  return ConfigurationTree(_backend->loadFile(path));
}


void ConfigurationTree::merge(ConfigurationTree conf){
  _backend->merge(conf._backend.get());
}
