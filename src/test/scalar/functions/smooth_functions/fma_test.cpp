#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class fma_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fma(nomad::tests::construct_unsafe_var<T>(x[0]),
               nomad::tests::construct_unsafe_var<T>(x[1]),
               nomad::tests::construct_unsafe_var<T>(x[2]));
  }
  static std::string name() { return "fma"; }
};


template <typename T>
class fma_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fma(T(x[0]), T(x[1]), T(x[2]));
  }
  static std::string name() { return "fma"; }
};

TEST(ScalarSmoothFunctions, Fma) {
  
  nomad::eigen_idx_t d = 3;
  
  Eigen::VectorXd x(d);
  x[0] = -2.483;
  x[1] = 0.576;
  x[2] = 1.384;
  
  nomad::tests::test_validation<fma_eval_func>(x);
  nomad::tests::test_derivatives<fma_grad_func>(x);
}

