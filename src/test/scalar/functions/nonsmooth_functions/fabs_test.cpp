#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class fabs_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fabs(nomad::tests::construct_unsafe_var<T>(x[0]));
  }
  static std::string name() { return "fabs"; }
};

template <typename T>
class fabs_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fabs(T(x[0]));
  }
  static std::string name() { return "fabs"; }
};

TEST(ScalarSmoothFunctions, Fabs) {
  
  nomad::eigen_idx_t d = 1;
  
  Eigen::VectorXd x(d);
  x[0] = 0.576;
  
  nomad::tests::test_validation<fabs_eval_func>(x);
  nomad::tests::test_derivatives<fabs_grad_func>(x);
}

