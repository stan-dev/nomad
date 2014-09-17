#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class erf_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return erf(nomad::tests::construct_unsafe_var<T>(x[0]));
  }
  static std::string name() { return "erf"; }
};

template <typename T>
class erf_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return erf(T(x[0]));
  }
  static std::string name() { return "erf"; }
};

TEST(ScalarSmoothFunctions, Erf) {
  
  nomad::eigen_idx_t d = 1;
  
  Eigen::VectorXd x(d);
  x[0] = 0.576;
  
  nomad::tests::test_validation<erf_eval_func>(x);
  nomad::tests::test_derivatives<erf_grad_func>(x);
}

