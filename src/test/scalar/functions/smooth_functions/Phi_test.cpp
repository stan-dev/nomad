#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class phi_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return Phi(nomad::tests::construct_unsafe_var<T>(x[0]));
  }
  static std::string name() { return "Phi"; }
};

template <typename T>
class phi_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return Phi(T(x[0]));
  }
  static std::string name() { return "Phi"; }
};

TEST(ScalarSmoothFunctions, Phi) {
  
  nomad::eigen_idx_t d = 1;
  
  Eigen::VectorXd x(d);
  x[0] = 1.576;
  
  nomad::tests::test_validation<phi_eval_func>(x);
  
  nomad::tests::test_derivatives<phi_grad_func>(x);
  
  x[0] = -60;
  nomad::tests::test_derivatives<phi_grad_func>(x);

  x[0] = -1.576;
  nomad::tests::test_derivatives<phi_grad_func>(x);

  x[0] = 60;
  nomad::tests::test_derivatives<phi_grad_func>(x);
  
}