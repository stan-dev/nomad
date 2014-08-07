#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

// clang++ -O3 -std=c++11 -I/Users/Betancourt/Documents/Research/Code/stan-dev/nomad -I/usr/local/include -I/Users/Betancourt/Documents/Research/Code/stan-dev/cmdstan/stan/lib/gtest_1.7.0/include -L/Users/Betancourt/Documents/Research/Code/stan-dev/cmdstan/stan/test -lgtest -o acos_test acos_test.cpp /Users/Betancourt/Documents/Research/Code/stan-dev/cmdstan/stan/lib/gtest_1.7.0/src/gtest_main.cc

template <typename T>
class acos_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return acos(v);
    
  }
  static std::string name() { return "acos"; }
};

TEST(ScalarSmoothFunctions, Acos) {
  Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
  x *= 0.576;
  nomad::tests::test_function<true, acos_func>(x);
}
