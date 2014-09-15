#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class log1p_exp_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return log1p_exp(v);
    
  }
  static std::string name() { return "log1p_exp"; }
};

TEST(ScalarSmoothFunctions, Log1pExp) {
  Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
  x *= 0.576;
  nomad::tests::test_function<true, false, log1p_exp_func>(x);
  x *= -1.0;
  nomad::tests::test_function<true, false, log1p_exp_func>(x);
}

