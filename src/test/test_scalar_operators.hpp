#ifndef nomad__src__tests__test_scalar_operators_hpp
#define nomad__src__tests__test_scalar_operators_hpp

#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/operators.hpp>
#include <src/tests/finite_difference.hpp>

namespace nomad {

  // Prototypes
  void test_operator_addition_assignment();
  void test_operator_addition();
  void test_operator_division_assignment();
  void test_operator_division();
  void test_operator_multiplication_assignment();
  void test_operator_multiplication();
  void test_operator_subtraction_assignment();
  void test_operator_subtraction();
  void test_operator_unary_decrement();
  void test_operator_unary_increment();
  void test_operator_unary_minus();
  void test_operator_unary_plus();
  
  void test_scalar_operators() {
    test_operator_addition_assignment();
    test_operator_addition();
    test_operator_division_assignment();
    test_operator_division();
    test_operator_multiplication_assignment();
    test_operator_multiplication();
    test_operator_subtraction_assignment();
    test_operator_subtraction();
    test_operator_unary_decrement();
    test_operator_unary_increment();
    test_operator_unary_minus();
    test_operator_unary_plus();
  }

  // operator_addition_assignment
  template <typename T>
  class operator_addition_assignment_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return exp(v1 += v2);
    }
    static std::string name() { return "operator_addition_assignment_vv"; }
  };
  
  template <typename T>
  class operator_addition_assignment_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v + 0.4847);
      
    }
    static std::string name() { return "operator_addition_assignment_vd"; }
  };
  
  void test_operator_addition_assignment() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 0.576;
    x1[1] *= -0.294;
    
    tests::test_function<operator_addition_assignment_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    x2 *= 0.576;
    
    tests::test_function<operator_addition_assignment_vd_func>(x2);
  }
  
  // operator_addition
  template <typename T>
  class operator_addition_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return exp(v1 + v2);
      
    }
    static std::string name() { return "operator_addition_vv"; }
  };
  
  template <typename T>
  class operator_addition_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v + 0.4847);
      
    }
    static std::string name() { return "operator_addition_vd"; }
  };
  
  template <typename T>
  class operator_addition_dv_func: public base_functor<T> {
  public:
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
  class operator_division_assignment_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return exp(v1 /= v2);
    }
    static std::string name() { return "operator_division_assignment_vv"; }
  };
  
  template <typename T>
  class operator_division_assignment_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v /= 0.4847);
      
    }
    static std::string name() { return "operator_division_assignment_vd"; }
  };
  
  void test_operator_division_assignment() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 0.576;
    x1[1] *= -0.294;
    
    tests::test_function<operator_division_assignment_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    x2 *= 0.576;
    
    tests::test_function<operator_division_assignment_vd_func>(x2);
  }
  
  // operator_division
  template <typename T>
  class operator_division_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return exp(v1 / v2);
      
    }
    static std::string name() { return "operator_division_vv"; }
  };
  
  template <typename T>
  class operator_division_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v / 0.4847);
      
    }
    static std::string name() { return "operator_division_vd"; }
  };
  
  template <typename T>
  class operator_division_dv_func: public base_functor<T> {
  public:
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
  
  // operator_multiplication_assignment
  template <typename T>
  class operator_multiplication_assignment_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return exp(v1 *= v2);
    }
    static std::string name() { return "operator_multiplication_assignment_vv"; }
  };
  
  template <typename T>
  class operator_multiplication_assignment_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v *= 0.4847);
      
    }
    static std::string name() { return "operator_multiplication_assignment_vd"; }
  };
  
  void test_operator_multiplication_assignment() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 0.576;
    x1[1] *= -0.294;
    
    tests::test_function<operator_multiplication_assignment_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    x2 *= 0.576;
    
    tests::test_function<operator_multiplication_assignment_vd_func>(x2);
  }
  
  // operator_multiplication
  template <typename T>
  class operator_multiplication_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return exp(v1 * v2);
      
    }
    static std::string name() { return "operator_multiplication_vv"; }
  };
  
  template <typename T>
  class operator_multiplication_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v * 0.4847);
      
    }
    static std::string name() { return "operator_multiplication_vd"; }
  };
  
  template <typename T>
  class operator_multiplication_dv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(0.3898 * v);
      
    }
    static std::string name() { return "operator_multiplication_dv"; }
  };
  
  void test_operator_multiplication() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 0.576;
    x1[1] *= -0.294;
    
    tests::test_function<operator_multiplication_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    x2 *= 0.576;
    
    tests::test_function<operator_multiplication_vd_func>(x2);
    tests::test_function<operator_multiplication_dv_func>(x2);
  }
  
  // operator_subtraction_assignment
  template <typename T>
  class operator_subtraction_assignment_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return exp(v1 -= v2);
    }
    static std::string name() { return "operator_subtraction_assignment_vv"; }
  };
  
  template <typename T>
  class operator_subtraction_assignment_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v -= 0.4847);
      
    }
    static std::string name() { return "operator_subtraction_assignment_vd"; }
  };
  
  void test_operator_subtraction_assignment() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 0.576;
    x1[1] *= -0.294;
    
    tests::test_function<operator_subtraction_assignment_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    x2 *= 0.576;
    
    tests::test_function<operator_subtraction_assignment_vd_func>(x2);
  }
  
  // operator_subtraction
  template <typename T>
  class operator_subtraction_vv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return exp(v1 - v2);
      
    }
    static std::string name() { return "operator_subtraction_vv"; }
  };
  
  template <typename T>
  class operator_subtraction_vd_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(v - 0.4847);
      
    }
    static std::string name() { return "operator_subtraction_vd"; }
  };
  
  template <typename T>
  class operator_subtraction_dv_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return exp(0.3898 - v);
      
    }
    static std::string name() { return "operator_subtraction_dv"; }
  };
  
  void test_operator_subtraction() {
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
    x1[0] *= 0.576;
    x1[1] *= -0.294;
    
    tests::test_function<operator_subtraction_vv_func>(x1);
    
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
    x2 *= 0.576;
    
    tests::test_function<operator_subtraction_vd_func>(x2);
    tests::test_function<operator_subtraction_dv_func>(x2);
  }

  // operator_unary_decrement
  template <typename T>
  class operator_unary_decrement_prefix_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      return exp(++v1);
      
    }
    static std::string name() { return "operator_unary_decrement_prefix"; }
  };
  
  template <typename T>
  class operator_unary_decrement_postfix_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      return exp(v1++);
      
    }
    static std::string name() { return "operator_unary_decrement_postfix"; }
  };
  
  void test_operator_unary_decrement() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<operator_unary_decrement_prefix_func>(x);
    tests::test_function<operator_unary_decrement_postfix_func>(x);
  }
  
  // operator_unary_increment
  template <typename T>
  class operator_unary_increment_prefix_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      return exp(++v1);
      
    }
    static std::string name() { return "operator_unary_increment_prefix"; }
  };
  
  template <typename T>
  class operator_unary_increment_postfix_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      return exp(v1++);
      
    }
    static std::string name() { return "operator_unary_increment_postfix"; }
  };
  
  void test_operator_unary_increment() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<operator_unary_increment_prefix_func>(x);
    tests::test_function<operator_unary_increment_postfix_func>(x);
  }
  
  // operator_unary_minus
  template <typename T>
  class operator_unary_minus_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      return exp(-v1);
      
    }
    static std::string name() { return "operator_unary_minus"; }
  };
  
  void test_operator_unary_minus() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<operator_unary_minus_func>(x);
  }

  // operator_unary_plus
  template <typename T>
  class operator_unary_plus_func: public base_functor<T> {
  public:
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      return exp(+v1);
      
    }
    static std::string name() { return "operator_unary_plus"; }
  };
  
  void test_operator_unary_plus() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<operator_unary_plus_func>(x);
  }

  
}

#endif
