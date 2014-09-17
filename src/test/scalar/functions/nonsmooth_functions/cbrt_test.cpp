#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class cbrt_eval_func {
public:
  
  T operator()(const Eigen::VectorXd& x) const {
    return cbrt(nomad::tests::construct_unsafe_var<T>(x[0]));
  }
  
  static std::string name() { return "cbrt"; }
};

template <typename T>
class cbrt_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return cbrt(v);
  }
  
  static std::string name() { return "cbrt"; }
};

TEST(ScalarNonSmoothFunctions, Cbrt) {
  Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
  x *= 0.576;
  nomad::tests::test_validation<cbrt_eval_func>(x);
  nomad::tests::test_derivatives<cbrt_grad_func>(x);
}

