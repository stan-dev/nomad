#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/scalar/operators.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class operator_greater_than_or_equal_to_vv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = nomad::tests::construct_unsafe_var<T>(x[0]);
    T v2 = nomad::tests::construct_unsafe_var<T>(x[1]);
    if (v1 >= v2)
      return v1;
    else
      return v2;
  }
  static std::string name() { return "operator_greater_than_or_equal_to_vv"; }
};

template <typename T>
class operator_greater_than_or_equal_to_vd_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = nomad::tests::construct_unsafe_var<T>(x[0]);
    if (v >= x[1])
      return -v;
    else
      return v;
  }
  static std::string name() { return "operator_greater_than_or_equal_to_vd"; }
};

template <typename T>
class operator_greater_than_or_equal_to_dv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = nomad::tests::construct_unsafe_var<T>(x[1]);
    if (x[0] >= v)
      return -v;
    else
      return v;
  }
  static std::string name() { return "operator_greater_than_or_equal_to_dv"; }
};

template <typename T>
class operator_greater_than_or_equal_to_vv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    if (v1 >= v2)
      return exp(v1);
    else
      return exp(v2);
  }
  static std::string name() { return "operator_greater_than_or_equal_to_vv"; }
};

template <typename T>
class operator_greater_than_or_equal_to_vd_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    if (v >= 1.0)
      return exp(-v);
    else
      return exp(v);
  }
  static std::string name() { return "operator_greater_than_or_equal_to_vd"; }
};

template <typename T>
class operator_greater_than_or_equal_to_dv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    if (1.0 >= v)
      return exp(v);
    else
      return exp(-v);
  }
  static std::string name() { return "operator_greater_than_or_equal_to_dv"; }
};

TEST(ScalarNonSmoothOperators, OperatorGreaterThanOrEqualTo) {

  nomad::eigen_idx_t d = 2;
  
  Eigen::VectorXd x1(d);
  x1[0] = 1.5;
  x1[1] = 0.5;
  
  nomad::tests::test_validation<operator_greater_than_or_equal_to_vv_eval_func>(x1);
  nomad::tests::test_validation<operator_greater_than_or_equal_to_vd_eval_func>(x1);
  nomad::tests::test_validation<operator_greater_than_or_equal_to_dv_eval_func>(x1);
  
  nomad::tests::test_derivatives<operator_greater_than_or_equal_to_vv_grad_func>(x1);
  
  x1[0] = 0.5;
  x1[1] = 1.5;
  nomad::tests::test_derivatives<operator_greater_than_or_equal_to_vv_grad_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  
  x2[0] = 1.5;
  nomad::tests::test_derivatives<operator_greater_than_or_equal_to_vd_grad_func>(x2);
  nomad::tests::test_derivatives<operator_greater_than_or_equal_to_dv_grad_func>(x2);
  
  x2[0] = 0.5;
  nomad::tests::test_derivatives<operator_greater_than_or_equal_to_vd_grad_func>(x2);
  nomad::tests::test_derivatives<operator_greater_than_or_equal_to_dv_grad_func>(x2);
}
