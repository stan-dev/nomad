#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class tgamma_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return tgamma(v);
    
  }
  static std::string name() { return "tgamma"; }
};

TEST(ScalarSmoothFunctions, Tgamma) {
  Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
  x *= 1.2483;
  nomad::tests::test_derivatives<true, true, tgamma_func>(x);
}

