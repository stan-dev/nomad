#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class atanh_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return atanh(nomad::tests::construct_unsafe_var<T>(x[0]));
  }
  static std::string name() { return "atanh"; }
};

template <typename T>
class atanh_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return atanh(T(x[0]));
  }
  static std::string name() { return "atanh"; }
};

TEST(ScalarSmoothFunctions, Atanh) {
  
  nomad::eigen_idx_t d = 1;
  
  Eigen::VectorXd x(d);
  x[0] = 0.576;
  
  Eigen::MatrixXd x_bad(d, 2);
  x_bad(0, 0) = -1.5;
  x_bad(0, 1) = 1.5;
  
  nomad::tests::test_validation<atanh_eval_func>(x, x_bad);
  nomad::tests::test_derivatives<atanh_grad_func>(x);
}
