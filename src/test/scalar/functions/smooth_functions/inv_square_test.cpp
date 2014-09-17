#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class inv_square_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return inv_square(nomad::tests::construct_unsafe_var<T>(x[0]));
  }
  static std::string name() { return "inv_square"; }
};

template <typename T>
class inv_square_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return inv_square(T(x[0]));
  }
  static std::string name() { return "inv_square"; }
};

TEST(ScalarSmoothFunctions, Inv_square) {
  
  nomad::eigen_idx_t d = 1;
  
  Eigen::VectorXd x(d);
  x[0] = 0.576;
  
  nomad::tests::test_validation<inv_square_eval_func>(x);
  nomad::tests::test_derivatives<inv_square_grad_func>(x);
}