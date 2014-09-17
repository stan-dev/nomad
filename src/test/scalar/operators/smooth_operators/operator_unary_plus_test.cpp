#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/operators.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class operator_unary_plus_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return +nomad::tests::construct_unsafe_var<T>(x[0]);
    
  }
  static std::string name() { return "operator_unary_plus"; }
};

template <typename T>
class operator_unary_plus_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    return exp(+v1);
    
  }
  static std::string name() { return "operator_unary_plus"; }
};

TEST(ScalarSmoothOperators, OperatorUnaryPlus) {

  nomad::eigen_idx_t d = 1;
  
  Eigen::VectorXd x(d);
  x[0] = 0.576;
  
  nomad::tests::test_validation<operator_unary_plus_eval_func>(x);
  nomad::tests::test_derivatives<operator_unary_plus_grad_func>(x);
}

