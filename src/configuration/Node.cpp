#include "Node.hpp"
using namespace std;
#include "iostream"

#include "NodeMapper.hpp"

Conf::Node* Conf::loadTree(ConfigurationTree conf, string field){

  try{
    conf.requireFields({field});
    ConfigurationTree sub = conf.getChild(field);
    return loadTree(sub);
  }catch(ConfigurationException e){
    throw ConfigurationException(e, "Error loading " + field);
  }

}

Conf::Node* Conf::loadTree(ConfigurationTree conf){
  conf.requireFields({"class"});
  string c = conf.get<string>("class");
  try{
    Node* node = NodeMapper::loadNode(c);

    if(conf.hasField("include")){
      string path = conf.get<string>("include");
      ConfigurationTree inc = conf.loadFile(path);
      conf.merge(inc);
    }

    node->load(conf);
    return node;
  }catch(ConfigurationException e){
    throw ConfigurationException(e, "Error loading " + c);
  }
}


