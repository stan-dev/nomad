#ifndef nomad__tests__test_scalar_functions_hpp
#define nomad__tests__test_scalar_functions_hpp

#include <string>

#include <scalar/functions.hpp>
#include <tests/finite_difference.hpp>

namespace nomad {

  // acos
  template <typename T>
  struct acos_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return acos(v);
      
    }
    static std::string name() { return "acos"; }
  };
  
  void test_acos() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<acos_func>(x);
  }
  
  // acosh
  template <typename T>
  struct acosh_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return acosh(v);
      
    }
    static std::string name() { return "acosh"; }
  };
  
  void test_acosh() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<acosh_func>(x);
  }
  
  // asin
  template <typename T>
  struct asin_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return asin(v);
      
    }
    static std::string name() { return "asin"; }
  };
  
  void test_asin() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<asin_func>(x);
  }
  
  // asinh
  template <typename T>
  struct asinh_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return asinh(v);
      
    }
    static std::string name() { return "asinh"; }
  };
  
  void test_asinh() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<asinh_func>(x);
  }
  
  // atan
  template <typename T>
  struct atan_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return atan(v);
      
    }
    static std::string name() { return "atan"; }
  };
  
  void test_atan() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<atan_func>(x);
  }
  
  // atanh
  template <typename T>
  struct atanh_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v = x[0];
      return atanh(v);
      
    }
    static std::string name() { return "atanh"; }
  };
  
  void test_atanh() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    x *= 0.576;
    tests::test_function<atanh_func>(x);
  }
 
  // binary_prod_cubes
  template <typename T>
  struct binary_prod_cubes_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      return binary_prod_cubes(v1, v2);
      
    }
    static std::string name() { return "binary_prod_cubes"; }
  };
  
  void test_binary_prod_cubes() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(2);
    x[0] = 0.576;
    x[1] = 0.832;
    tests::test_function<binary_prod_cubes_func>(x);
  }
  
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
  
  // trinary_prod_cubes
  template <typename T>
  struct trinary_prod_cubes_func {
    T operator()(const Eigen::VectorXd& x) const {
      T v1 = x[0];
      T v2 = x[1];
      T v3 = x[2];
      return trinary_prod_cubes(v1, v2, v3);
      
    }
    static std::string name() { return "trinary_prod_cubes"; }
  };
  
  void test_trinary_prod_cubes() {
    Eigen::VectorXd x = Eigen::VectorXd::Ones(3);
    x[0] = 0.576;
    x[1] = 0.832;
    x[1] = -1.765;
    tests::test_function<trinary_prod_cubes_func>(x);
  }
  
  void test_scalar_functions() {
    test_acos();
    test_acosh();
    test_asin();
    test_asinh();
    test_atan();
    test_atanh();
    test_binary_prod_cubes();
    test_exp();
    test_square();
    test_trinary_prod_cubes();
  }

}

#endif
