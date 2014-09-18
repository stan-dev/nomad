#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/scalar/operators.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class if_else_vv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return if_else(x[0] > x[1],
                   nomad::tests::construct_unsafe_var<T>(x[0]),
                   nomad::tests::construct_unsafe_var<T>(x[1]));
  }
  static std::string name() { return "if_else_vv"; }
};

template <typename T>
class if_else_vd_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return if_else(x[0] > x[1], nomad::tests::construct_unsafe_var<T>(x[0]), x[1]);
  }
  static std::string name() { return "if_else_vd"; }
};

template <typename T>
class if_else_dv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return if_else(x[0] > x[1], x[0], nomad::tests::construct_unsafe_var<T>(x[1]));
  }
  static std::string name() { return "if_else_dv"; }
};

template <typename T>
class if_else_vv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return if_else(v1 > v2, exp(v1), exp(v2));
  }
  static std::string name() { return "if_else_vv"; }
};

template <typename T>
class if_else_vd_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return if_else(v > 0.5, exp(v), 0.5);
  }
  static std::string name() { return "if_else_vd"; }
};

template <typename T>
class if_else_dv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return if_else(0.5 > v, 0.5, exp(v));
  }
  static std::string name() { return "if_else_dv"; }
};

TEST(ScalarNonSmoothFunctions, IfElse) {

  nomad::eigen_idx_t d = 2;
  
  Eigen::VectorXd x1(d);
  x1[0] = 1.0;
  x1[1] = 0.5;

  nomad::tests::test_validation<if_else_vv_eval_func>(x1);
  nomad::tests::test_validation<if_else_vd_eval_func>(x1);
  nomad::tests::test_validation<if_else_dv_eval_func>(x1);
  
  nomad::tests::test_derivatives<if_else_vv_grad_func>(x1);
  
  x1 *= -1;
  nomad::tests::test_derivatives<if_else_vv_grad_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  
  x2[0] = 0.75;
  nomad::tests::test_derivatives<if_else_vd_grad_func>(x2);
  nomad::tests::test_derivatives<if_else_dv_grad_func>(x2);
  
  x2[0] = 0.25;
  nomad::tests::test_derivatives<if_else_vd_grad_func>(x2);
  nomad::tests::test_derivatives<if_else_dv_grad_func>(x2);
}
