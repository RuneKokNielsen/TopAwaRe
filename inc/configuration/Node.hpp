#ifndef NODE_HPP
#define NODE_HPP

#include "ConfigurationTree.hpp"

/**
 * Abstract base element for computation tree. Takes a configuration (sub)tree
 * and loads the computational (sub)tree
 **/
namespace Conf{

class Node{

public:
  /**
   * Each class inheriting Node should implement how to load
   * based on an input configuration subtree.
   * \param conf The subtree corresponding to this node.
   **/
  virtual void load(ConfigurationTree conf) = 0;
  virtual ~Node(){};

};

  /**
   * Given a tree, load and return the node corresponding
   * to the subtree by the given name.
   * \param conf The current tree.
   * \param field The name of the subtree to load.
   **/
  Node* loadTree(ConfigurationTree conf, string field);
  /**
   * Given a tree, recursively load and return its root node.
   * \param conf The tree to load.
   **/
  Node* loadTree(ConfigurationTree conf);


}

#endif
