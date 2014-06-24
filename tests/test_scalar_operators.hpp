#ifndef nomad__tests__test_scalar_operators_hpp
#define nomad__tests__test_scalar_operators_hpp

#include <string>

#include <scalar/operators.hpp>
#include <tests/finite_difference.hpp>

namespace nomad {

  // operator_divide
  template <typename T>
  struct operator_divide_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v) / 0.588;
      
    }
    static std::string name() { return "operator_divide"; }
  };
  
  void test_operator_divide() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<operator_divide_func>(x);
  }
  
  // operator_multiply
  template <typename T>
  struct operator_multiply_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return v1 * exp(v1) * exp(v2) * v2;
      
    }
    static std::string name() { return "operator_multiply"; }
  };
  
  void test_operator_multiply() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(2);
    x *= 0.576;
    tests::test_function<operator_multiply_func>(x);
  }
  
  // operator plus_equals
  template <typename T>
  struct operator_plus_equals_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      
      T v3 = exp(v1);
      v3 += exp(v2);
      
      return v3;
      
    }
    static std::string name() { return "operator_plus_equals"; }
  };
  
  void test_operator_plus_equals() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(2);
    x *= 0.576;
    tests::test_function<operator_plus_equals_func>(x);
  }
  
  // operator plus
  template <typename T>
  struct operator_plus_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return exp(v1) + exp(v2);
      
    }
    static std::string name() { return "operator_plus"; }
  };
  
  void test_operator_plus() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(2);
    x *= 0.576;
    tests::test_function<operator_plus_func>(x);
  }
  
  // operator unary_negative
  template <typename T>
  struct operator_unary_negative_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      return - exp(v1);
      
    }
    static std::string name() { return "operator_unary_negative"; }
  };
  
  void test_operator_unary_negative() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<operator_unary_negative_func>(x);
  }
  
  void test_scalar_operators() {
    test_operator_divide();
    test_operator_multiply();
    test_operator_plus_equals();
    test_operator_plus();
    test_operator_unary_negative();
  }

}

#endif
