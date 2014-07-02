#ifndef nomad__tests__test_scalar_operators_hpp
#define nomad__tests__test_scalar_operators_hpp

#include <string>

#include <scalar/operators.hpp>
#include <tests/finite_difference.hpp>

namespace nomad {

  // Prototypes
  void test_operator_addition_assignment();
  void test_operator_addition();
  void test_operator_division_assignment();
  void test_operator_division();
  void test_operator_multiply();
  void test_operator_plus_equals();
  void test_operator_unary_negative();
  
  void test_scalar_operators() {
    test_operator_addition_assignment();
    test_operator_addition();
    test_operator_division_assignment();
    test_operator_division();
    test_operator_multiply();
    test_operator_unary_negative();
  }

  // operator_addition_assignment
  template <typename T>
  struct operator_addition_assignment_vv_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return exp(v1 += v2);
    }
    static std::string name() { return "operator_addition_assignment_vv"; }
  };
  
  template <typename T>
  struct operator_addition_assignment_vd_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v + 0.4847);
      
    }
    static std::string name() { return "operator_addition_assignment_vd"; }
  };
  
  template <typename T>
  struct operator_addition_assignment_dv_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(0.3898 + v);
      
    }
    static std::string name() { return "operator_addition_assignment_dv"; }
  };
  
  void test_operator_addition_assignment() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 0.576;
    x1[1] *= -0.294;
    
    tests::test_function<operator_addition_assignment_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    x2 *= 0.576;
    
    tests::test_function<operator_addition_assignment_vd_func>(x2);
    tests::test_function<operator_addition_assignment_dv_func>(x2);
  }
  
  // operator_addition
  template <typename T>
  struct operator_addition_vv_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return exp(v1 + v2);
      
    }
    static std::string name() { return "operator_addition_vv"; }
  };
  
  template <typename T>
  struct operator_addition_vd_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v + 0.4847);
      
    }
    static std::string name() { return "operator_addition_vd"; }
  };
  
  template <typename T>
  struct operator_addition_dv_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(0.3898 + v);
      
    }
    static std::string name() { return "operator_addition_dv"; }
  };
  
  void test_operator_addition() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 0.576;
    x1[1] *= -0.294;
    
    tests::test_function<operator_addition_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    x2 *= 0.576;
    
    tests::test_function<operator_addition_vd_func>(x2);
    tests::test_function<operator_addition_dv_func>(x2);
  }

  // operator_division_assignment
  template <typename T>
  struct operator_division_assignment_vv_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return exp(v1 /= v2);
    }
    static std::string name() { return "operator_division_assignment_vv"; }
  };
  
  template <typename T>
  struct operator_division_assignment_vd_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v / 0.4847);
      
    }
    static std::string name() { return "operator_division_assignment_vd"; }
  };
  
  template <typename T>
  struct operator_division_assignment_dv_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(0.3898 / v);
      
    }
    static std::string name() { return "operator_division_assignment_dv"; }
  };
  
  void test_operator_division_assignment() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 0.576;
    x1[1] *= -0.294;
    
    tests::test_function<operator_division_assignment_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    x2 *= 0.576;
    
    tests::test_function<operator_division_assignment_vd_func>(x2);
    tests::test_function<operator_division_assignment_dv_func>(x2);
  }
  
  // operator_division
  template <typename T>
  struct operator_division_vv_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return v1 / v2;
      
    }
    static std::string name() { return "operator_division_vv"; }
  };
  
  template <typename T>
  struct operator_division_vd_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v / 0.4847);
      
    }
    static std::string name() { return "operator_division_vd"; }
  };
  
  template <typename T>
  struct operator_division_dv_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(0.3898 / v);
      
    }
    static std::string name() { return "operator_division_dv"; }
  };
  
  void test_operator_division() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 0.576;
    x1[1] *= -0.294;
    
    tests::test_function<operator_division_vv_func>(x1);

    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    x2 *= 0.576;
    
    tests::test_function<operator_division_vd_func>(x2);
    tests::test_function<operator_division_dv_func>(x2);
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

}

#endif
