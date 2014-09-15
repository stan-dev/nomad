#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class lgamma_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return lgamma(v);
    
  }
  static std::string name() { return "lgamma"; }
};

TEST(ScalarSmoothFunctions, Lgamma) {
  Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
  x *= 4.584;
  nomad::tests::test_function<true, false, lgamma_func>(x);
}

