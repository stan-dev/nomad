#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class fma_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    T v3 = x[2];
    return fma(v1, v2, v3);
    
  }
  static std::string name() { return "fma"; }
};

TEST(ScalarSmoothFunctions, Fma) {
  Eigen::VectorXd x = Eigen::VectorXd::Ones(3);
  x[0] *= -2.483;
  x[1] *= 0.576;
  x[2] *= 1.384;
  nomad::tests::test_derivatives<true, true, fma_func>(x);
}

