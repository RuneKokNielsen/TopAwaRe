#ifndef CONFIGURATION_TREE_HPP
#define CONFIGURATION_TREE_HPP

using namespace std;
#include <string>
#include <vector>
#include <memory>

/**
 * Exception signifying faults in the configuration contents.
 **/
struct ConfigurationException: public exception{
  std::string _str;
  ConfigurationException(std::string str) : _str(str) {}
  ConfigurationException(ConfigurationException src, string str){
    _str = str + " -> " + src.what();
  }
  ~ConfigurationException() throw () {}
  const char* what() const throw() { return _str.c_str(); }
};

/**
 * A ConfigurationTreeBackend is the underlying implementation of the
 * ConfigurationTree e.g. based on JSON format.
 **/
class ConfigurationTreeBackend{

public:

  virtual ~ConfigurationTreeBackend(){};

  /**
   * Get string value by field name.
   * \param field field name.
   **/
  virtual string getString(string field) = 0;
  /**
   * Get float value by field name.
   * \param field field name.
   **/
  virtual float getFloat(string field) = 0;
  /**
   * Get vector of floats by field name.
   * \param field field name.
   **/
  virtual vector<float> getVecFloat(string field) = 0;
  /**
   * Get 2D vector of floats value by field name.
   * \param field field name.
   **/
  virtual vector<vector<float>> getVecVecFloat(string field) = 0;
  /**
   * Get 3D vector of floats value by field name.
   * \param field field name.
   **/
  virtual vector<vector<vector<float>>> getVecVecVecFloat(string field) = 0;
  /**
   * Get integer value by field name.
   * \param field field name.
   **/
  virtual int getInt(string field) = 0;
  /**
   * Get vector of integers value by field name.
   * \param field field name.
   **/
  virtual vector<int> getVecInt(string field) = 0;
  /**
   * Get boolean value by field name.
   * \param field field name.
   **/
  virtual bool getBool(string field) = 0;

  /**
   * Get interface to subtree by field name.
   * \param field field name.
   **/
  virtual ConfigurationTreeBackend* getChild(string field) = 0;
  /**
   * Get vector of interfaces to subtrees belonging to same field by field name.
   * \param field field name.
   **/
  virtual vector<ConfigurationTreeBackend*> getChildren(string field) = 0;

  /**
   * Returns whether field exists.
   * \param field field name.
   **/
  virtual bool hasField(string field) = 0;

  /**
   * Load configuration tree backend from data in file by path.
   * \param path path to file.
   **/
  virtual ConfigurationTreeBackend* loadFile(string path) = 0;

  /**
   * Merge the key-value pairs from given configuration tree backend
   * into this configuration tree backend instance.
   * \param conf Configuration tree to merge into this.
   **/
  virtual void merge(ConfigurationTreeBackend* conf) = 0;
};

/**
 * Configuration tree as loaded from some configuration.
 * Works as an interface to some ConfigurationTreeBackend.
 **/
class ConfigurationTree{

public:

  /**
   * Instantiate a ConfigurationTree through some backend.
   * \param backend The backend interfacing to the configuration data.
   **/
  ConfigurationTree(ConfigurationTreeBackend* backend){
    _backend = shared_ptr<ConfigurationTreeBackend>(backend);
  };
  ~ConfigurationTree(){
  }

  /**
   * Checks that all required fields are present at this level
   * of the configuration.
   * \param fields Vector of required fields.
   **/
  void requireFields(vector<string> fields);

  /**
   * Templated field value getter. Attempts to map the typename to
   * a supported type in the ConfigurationTreeBackend.
   * \param field field name.
   **/
  template<typename T> T get(string field);
  /**
   * Same as get(string field), but returns a given default value
   * if the given field is not present.
   * \param field field name.
   * \param dfault default value.
   **/
  template<typename T> T get(string field, T dfault){
    return hasField(field) ? get<T>(field) : dfault;
  }

  /**
   * Returns true if field with given name is present at this level
   * of the configuration.
   * \param field field name.
   **/
  bool hasField(string field);

  /**
   * Returns true if field with given name is present at this level
   * and is a boolean corresponding to the value true.
   * \param field field name.
   **/
  bool getFlag(string field);

  /**
   * Returns a subtree corresponding to the given field.
   * \param field field name of subtree.
   **/
  ConfigurationTree getChild(string field);

  /**
   * Returns a vector of subtrees under the same field name.
   * \param field field name.
   **/
  vector<ConfigurationTree> getChildren(string field);

  /**
   * Loads a configuration tree from a file.
   * \param path path to file.
   **/
  ConfigurationTree loadFile(string path);

  /**
   * Merges another configuration tree into the root
   * of this one.
   * \param conf The other configuration tree.
   **/
  void merge(ConfigurationTree conf);

private:

  /**
   * The backend used to interface with the configuration data.
   **/
  shared_ptr<ConfigurationTreeBackend> _backend;

};





#endif
