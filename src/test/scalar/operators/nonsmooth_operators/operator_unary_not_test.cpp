#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/scalar/operators.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class operator_unary_not_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = nomad::tests::construct_unsafe_var<T>(x[0]);
    if (!v)
      return -v;
    else
      return v;
  }
  static std::string name() { return "operator_unary_not"; }
};

TEST(ScalarNonSmoothOperators, OperatorUnaryNot) {
  
  nomad::eigen_idx_t d = 1;
  Eigen::VectorXd x(d);
  x[0] = 1.5;
  
  nomad::tests::test_validation<operator_unary_not_eval_func>(x);
  
}
