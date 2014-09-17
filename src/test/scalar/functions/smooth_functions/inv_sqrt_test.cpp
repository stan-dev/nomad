#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class inv_sqrt_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return inv_sqrt(nomad::tests::construct_unsafe_var<T>(x[0]));
  }
  static std::string name() { return "inv_sqrt"; }
};

template <typename T>
class inv_sqrt_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return inv_sqrt(T(x[0]));
  }
  static std::string name() { return "inv_sqrt"; }
};

TEST(ScalarSmoothFunctions, Inv_sqrt) {
  
  nomad::eigen_idx_t d = 1;
  
  Eigen::VectorXd x(d);
  x[0] = 0.576;
  
  Eigen::MatrixXd x_bad(d, 1);
  x_bad(0, 0) = -1.5;
  
  nomad::tests::test_validation<inv_sqrt_eval_func>(x, x_bad);
  nomad::tests::test_derivatives<inv_sqrt_grad_func>(x);
}
