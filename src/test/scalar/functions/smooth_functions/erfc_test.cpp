#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class erfc_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return erfc(nomad::tests::construct_unsafe_var<T>(x[0]));
  }
  static std::string name() { return "erfc"; }
};

template <typename T>
class erfc_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return erfc(T(x[0]));
  }
  static std::string name() { return "erfc"; }
};

TEST(ScalarSmoothFunctions, Erfc) {
  
  nomad::eigen_idx_t d = 1;
  
  Eigen::VectorXd x(d);
  x[0] = 0.576;
  
  nomad::tests::test_validation<erfc_eval_func>(x);
  nomad::tests::test_derivatives<erfc_grad_func>(x);
}
