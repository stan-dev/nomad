#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class expm1_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return expm1(v);
    
  }
  static std::string name() { return "expm1"; }
};

TEST(ScalarSmoothFunctions, Expm1) {
  Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
  x *= 0.576;
  nomad::tests::test_derivatives<true, true, expm1_func>(x);
}

