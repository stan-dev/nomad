#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/operators.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class operator_multiplication_vv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return nomad::tests::construct_unsafe_var<T>(x[0]) * nomad::tests::construct_unsafe_var<T>(x[1]);
  }
  static std::string name() { return "operator_multiplication_vv"; }
};

template <typename T>
class operator_multiplication_vd_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return nomad::tests::construct_unsafe_var<T>(x[0]) * x[1];
    
  }
  static std::string name() { return "operator_multiplication_vd"; }
};

template <typename T>
class operator_multiplication_dv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return x[0] * nomad::tests::construct_unsafe_var<T>(x[1]);
    
  }
  static std::string name() { return "operator_multiplication_vd"; }
};

template <typename T>
class operator_multiplication_vv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return exp(v1 * v2);
  }
  static std::string name() { return "operator_multiplication_vv"; }
};

template <typename T>
class operator_multiplication_vd_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return exp(v * 0.4847);
    
  }
  static std::string name() { return "operator_multiplication_vd"; }
};

template <typename T>
class operator_multiplication_dv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return exp(0.4847 * v);
    
  }
  static std::string name() { return "operator_multiplication_vd"; }
};

TEST(ScalarSmoothOperators, OperatorMultiplication) {
  nomad::eigen_idx_t d = 2;
  
  Eigen::VectorXd x1(d);
  x1[0] = 0.576;
  x1[1] = -0.294;
  
  nomad::tests::test_validation<operator_multiplication_vv_eval_func>(x1);
  nomad::tests::test_validation<operator_multiplication_vd_eval_func>(x1);
  nomad::tests::test_validation<operator_multiplication_dv_eval_func>(x1);
  
  nomad::tests::test_derivatives<operator_multiplication_vv_grad_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  x2 *= 0.576;
  
  nomad::tests::test_derivatives<operator_multiplication_vd_grad_func>(x2);
  nomad::tests::test_derivatives<operator_multiplication_dv_grad_func>(x2);
}

