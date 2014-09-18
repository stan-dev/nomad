#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/scalar/operators.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class operator_not_equal_to_vv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = nomad::tests::construct_unsafe_var<T>(x[0]);
    T v2 = nomad::tests::construct_unsafe_var<T>(x[1]);
    if (v1 != v2)
      return v1;
    else
      return v2;
  }
  static std::string name() { return "operator_not_equal_to_vv"; }
};

template <typename T>
class operator_not_equal_to_vd_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = nomad::tests::construct_unsafe_var<T>(x[0]);
    if (v != x[1])
      return -v;
    else
      return v;
  }
  static std::string name() { return "operator_not_equal_to_vd"; }
};

template <typename T>
class operator_not_equal_to_dv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = nomad::tests::construct_unsafe_var<T>(x[1]);
    if (x[0] != v)
      return -v;
    else
      return v;
  }
  static std::string name() { return "operator_not_equal_to_dv"; }
};


TEST(ScalarNonSmoothOperators, OperatorNotEqualTo) {
  
  nomad::eigen_idx_t d = 2;
  
  Eigen::VectorXd x1(d);
  x1[0] = 1.5;
  x1[1] = 0.5;
  
  nomad::tests::test_validation<operator_not_equal_to_vv_eval_func>(x1);
  nomad::tests::test_validation<operator_not_equal_to_vd_eval_func>(x1);
  nomad::tests::test_validation<operator_not_equal_to_dv_eval_func>(x1);
  
}
