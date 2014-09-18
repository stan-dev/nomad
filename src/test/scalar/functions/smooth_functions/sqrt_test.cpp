#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class sqrt_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return sqrt(nomad::tests::construct_unsafe_var<T>(x[0]));
  }
  static std::string name() { return "sqrt"; }
};

template <typename T>
class sqrt_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return sqrt(T(x[0]));
  }
  static std::string name() { return "sqrt"; }
};

TEST(ScalarSmoothFunctions, Sqrt) {
  
  nomad::eigen_idx_t d = 1;
  
  Eigen::VectorXd x(d);
  x[0] = 1.576;
  
  Eigen::MatrixXd x_bad(d, 1);
  x_bad(0, 0) = -1;
  
  nomad::tests::test_validation<sqrt_eval_func>(x, x_bad);
  nomad::tests::test_derivatives<sqrt_grad_func>(x);
}
