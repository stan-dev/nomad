#ifndef nomad__tests__scalar__functions__smooth_functions__acos_hpp
#define nomad__tests__scalar__functions__smooth_functions__acos_hpp

#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <autodiff/base_functor.hpp>
#include <scalar/functions.hpp>
#include <tests/finite_difference.hpp>

namespace nomad {
  namespace tests {

    template <typename T>
    class acos_func: public base_functor<T> {
    public:
      T operator()(const Eigen::VectorXd& x) const {
        T v = x[0];
        return acos(v);
        
      }
      static std::string name() { return "acos"; }
    };
    
  }
}

TEST(ScalarSmoothFunctions, Acos) {
  Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
  x *= 0.576;
  nomad::tests::test_function<true, nomad::tests::acos_func>(x);
}

#endif
