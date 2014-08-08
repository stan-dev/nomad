#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class multiply_log_vv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return multiply_log(v1, v2);
    
  }
  static std::string name() { return "multiply_log_vv"; }
};

template <typename T>
class multiply_log_vd_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return multiply_log(v, 0.5);
    
  }
  static std::string name() { return "multiply_log_vd"; }
};

template <typename T>
class multiply_log_dv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return multiply_log(0.5, v);
    
  }
  static std::string name() { return "multiply_log_dv"; }
};

TEST(ScalarSmoothFunctions, MultiplyLog) {
  Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
  x1[0] *= 1.0;
  x1[1] *= 0.5;
  
  nomad::tests::test_function<true, multiply_log_vv_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  
  nomad::tests::test_function<true, multiply_log_vd_func>(x2);
  nomad::tests::test_function<true, multiply_log_dv_func>(x2);
}

