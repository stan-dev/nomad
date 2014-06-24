#ifndef nomad__tests__test_scalar_functions_hpp
#define nomad__tests__test_scalar_functions_hpp

#include <string>

#include <scalar/functions.hpp>
#include <tests/finite_difference.hpp>

namespace nomad {

  // exp
  template <typename T>
  struct exp_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v);
      
    }
    static std::string name() { return "exp"; }
  };
  
  void test_exp() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<exp_func>(x);
  }
  
  // square
  template <typename T>
  struct square_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return square(exp(v));
      
    }
    static std::string name() { return "square"; }
  };
  
  void test_square() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<square_func>(x);
  }
  
  void test_scalar_functions() {
    test_exp();
    test_square();
  }

}

#endif
