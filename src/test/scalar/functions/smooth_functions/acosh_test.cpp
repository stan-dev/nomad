#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class acosh_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return acosh(v);
  }
  static std::string name() { return "acosh"; }
};

TEST(ScalarSmoothFunctions, Acosh) {
  Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
  x *= 1.576;
  nomad::tests::test_function<true, acosh_func>(x);
}

