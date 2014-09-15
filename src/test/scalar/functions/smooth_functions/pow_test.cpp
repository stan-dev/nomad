#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class pow_vv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return pow(v1, v2);
    
  }
  static std::string name() { return "pow_vv"; }
};

template <typename T>
class pow_vd_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return pow(v, 0.4847);
    
  }
  static std::string name() { return "pow_vd"; }
};

template <typename T>
class pow_dv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return pow(0.3898, v);
    
  }
  static std::string name() { return "pow_dv"; }
};

TEST(ScalarSmoothFunctions, Pow) {
  Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
  x1[0] *= 0.576;
  x1[1] *= -0.294;
  
  nomad::tests::test_function<true, false, pow_vv_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  x2 *= 0.576;
  
  nomad::tests::test_function<true, false, pow_vd_func>(x2);
  nomad::tests::test_function<true, false, pow_dv_func>(x2);
}

