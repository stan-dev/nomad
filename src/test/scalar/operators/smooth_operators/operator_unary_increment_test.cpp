#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/operators.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class operator_unary_increment_prefix_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    return exp(++v1);
  }
  static std::string name() { return "operator_unary_increment_prefix"; }
};

template <typename T>
class operator_unary_increment_postfix_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    return exp(v1++);
  }
  static std::string name() { return "operator_unary_increment_postfix"; }
};

TEST(ScalarSmoothOperators, OperatorUnaryIncrement) {
  Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
  x *= 0.576;
  nomad::tests::test_derivatives<true, true, operator_unary_increment_prefix_func>(x);
  nomad::tests::test_derivatives<true, true, operator_unary_increment_postfix_func>(x);
}

