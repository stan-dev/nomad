#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class multiply_log_vv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return multiply_log(nomad::tests::construct_unsafe_var<T>(x[0]),
                        nomad::tests::construct_unsafe_var<T>(x[1]));
    
  }
  static std::string name() { return "multiply_log_vv"; }
};

template <typename T>
class multiply_log_vd_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return multiply_log(nomad::tests::construct_unsafe_var<T>(x[0]),
                        x[1]);
    
  }
  static std::string name() { return "multiply_log_vd"; }
};

template <typename T>
class multiply_log_dv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return multiply_log(x[0],
                        nomad::tests::construct_unsafe_var<T>(x[1]));
    
  }
  static std::string name() { return "multiply_log_dv"; }
};

template <typename T>
class multiply_log_vv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return multiply_log(T(x[0]), T(x[1]));
    
  }
  static std::string name() { return "multiply_log_vv"; }
};

template <typename T>
class multiply_log_vd_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return multiply_log(T(x[0]), 0.5);
    
  }
  static std::string name() { return "multiply_log_vd"; }
};

template <typename T>
class multiply_log_dv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return multiply_log(0.5, T(x[0]));
    
  }
  static std::string name() { return "multiply_log_dv"; }
};

TEST(ScalarSmoothFunctions, MultiplyLog) {
  
  nomad::eigen_idx_t d = 2;
  
  Eigen::VectorXd x1(d);
  x1[0] = 1.0;
  x1[1] = 0.5;
  
  Eigen::MatrixXd x1_bad(d, 1);
  x1_bad(0, 0) = 0.5;
  x1_bad(1, 0) = -1.0;
  
  nomad::tests::test_validation<multiply_log_vv_eval_func>(x1, x1_bad);
  nomad::tests::test_validation<multiply_log_vd_eval_func>(x1, x1_bad);
  nomad::tests::test_validation<multiply_log_dv_eval_func>(x1, x1_bad);
  
  nomad::tests::test_derivatives<multiply_log_vv_grad_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  
  nomad::tests::test_derivatives<multiply_log_vd_grad_func>(x2);
  nomad::tests::test_derivatives<multiply_log_dv_grad_func>(x2);
}

