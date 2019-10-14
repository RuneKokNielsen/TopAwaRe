#include "JSONConfigurationTree.hpp"
#include <fstream>


JSONConfigurationTree* JSONConfigurationTree::getChild(string field){
  return new JSONConfigurationTree(_j[field].get<json>());
}


#include <iostream>
vector<ConfigurationTreeBackend*> JSONConfigurationTree::getChildren(string field){

  vector<ConfigurationTreeBackend*> res;
  for(uint i=0; i<_j[field].size(); i++){
    res.push_back(new JSONConfigurationTree(_j[field].at(i).get<json>()));
  }
  return res;  
}

string JSONConfigurationTree::getString(string field){
  return get<string>(field);
}

float JSONConfigurationTree::getFloat(string field){
  return get<float>(field);
}

int JSONConfigurationTree::getInt(string field){
  return get<int>(field);
}

vector<int> JSONConfigurationTree::getVecInt(string field){
  try{
    return get<vector<int> >(field);
  }catch(nlohmann::detail::type_error e){
    vector<int> v;
    v.push_back(get<int>(field));
    return v;
  }
}

bool JSONConfigurationTree::getBool(string field){
  return get<bool>(field);
}

vector<float> JSONConfigurationTree::getVecFloat(string field){
  try{
    return get<vector<float> >(field);
  }catch(nlohmann::detail::type_error e){
    vector<float> v;
    v.push_back(get<float>(field));
    return v;
  }  
}

vector<vector<float>> JSONConfigurationTree::getVecVecFloat(string field){
  return get<vector<vector<float> > >(field);
}

vector<vector<vector<float>>> JSONConfigurationTree::getVecVecVecFloat(string field){
  return get<vector<vector<vector<float>>>>(field);
}

bool JSONConfigurationTree::hasField(string field){

  return _j.find(field) != _j.end();
  
}



JSONConfigurationTree* JSONConfigurationTree::loadFile(string path){
  ifstream f(path);
  json j;
  f >> j;
  f.close();

  return new JSONConfigurationTree(j);
}


void JSONConfigurationTree::merge(ConfigurationTreeBackend* conf){
  JSONConfigurationTree* jconf = (JSONConfigurationTree*) conf;  
  for (json::iterator it = jconf->_j.begin(); it != jconf->_j.end(); ++it) {
    _j[it.key()] = it.value();
  }
}
