#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class log_diff_exp_vv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return log_diff_exp(v1, v2);
    
  }
  static std::string name() { return "log_diff_exp_vv"; }
};

template <typename T>
class log_diff_exp_vd_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return log_diff_exp(v, 0.5);
    
  }
  static std::string name() { return "log_diff_exp_vd"; }
};

template <typename T>
class log_diff_exp_dv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return log_diff_exp(0.5, v);
    
  }
  static std::string name() { return "log_diff_exp_dv"; }
};

TEST(ScalarSmoothFunctions, LogDiffExp) {
  Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
  x1[0] = 1.0;
  x1[1] = 0.5;
  
  nomad::tests::test_function<true, log_diff_exp_vv_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  
  nomad::tests::test_function<true, log_diff_exp_vd_func>(x2);
  
  x2 *= 0.25;
  nomad::tests::test_function<true, log_diff_exp_dv_func>(x2);

}

