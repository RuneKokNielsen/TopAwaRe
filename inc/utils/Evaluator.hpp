#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP



template<typename T1, typename T2>
class Evaluator{

public:
  virtual T2 eval(T1 input) = 0;

  virtual ~Evaluator(){};

};









#endif
