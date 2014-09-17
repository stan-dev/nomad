#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class log_sum_exp_vv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return log_sum_exp(nomad::tests::construct_unsafe_var<T>(x[0]),
                       nomad::tests::construct_unsafe_var<T>(x[1]));
    
  }
  static std::string name() { return "log_sum_exp_vv"; }
};

template <typename T>
class log_sum_exp_vd_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return log_sum_exp(nomad::tests::construct_unsafe_var<T>(x[0]),
                       x[1]);
    
  }
  static std::string name() { return "log_sum_exp_vd"; }
};

template <typename T>
class log_sum_exp_dv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return log_sum_exp(x[0],
                       nomad::tests::construct_unsafe_var<T>(x[1]));
    
  }
  static std::string name() { return "log_sum_exp_dv"; }
};

template <typename T>
class log_sum_exp_vv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return log_sum_exp(v1, v2);
    
  }
  static std::string name() { return "log_sum_exp_vv"; }
};

template <typename T>
class log_sum_exp_vd_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return log_sum_exp(T(x[0]), 0.5);
    
  }
  static std::string name() { return "log_sum_exp_vd"; }
};

template <typename T>
class log_sum_exp_dv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return log_sum_exp(0.5, T(x[0]));
    
  }
  static std::string name() { return "log_sum_exp_dv"; }
};

TEST(ScalarSmoothFunctions, LogSumExp) {
  
  nomad::eigen_idx_t d = 2;
  
  Eigen::VectorXd x1(d);
  x1[0] = 1.0;
  x1[1] = 0.5;
  
  nomad::tests::test_validation<log_sum_exp_vv_eval_func>(x1);
  nomad::tests::test_validation<log_sum_exp_vd_eval_func>(x1);
  nomad::tests::test_validation<log_sum_exp_dv_eval_func>(x1);
  
  nomad::tests::test_derivatives<log_sum_exp_vv_grad_func>(x1);
  
  x1[0] = -1.0;
  nomad::tests::test_derivatives<log_sum_exp_vv_grad_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  
  nomad::tests::test_derivatives<log_sum_exp_vd_grad_func>(x2);
  nomad::tests::test_derivatives<log_sum_exp_dv_grad_func>(x2);
  
  x2[0] = -1.0;
  nomad::tests::test_derivatives<log_sum_exp_vd_grad_func>(x2);
  nomad::tests::test_derivatives<log_sum_exp_dv_grad_func>(x2);
}

