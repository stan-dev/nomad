#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class hypot_vv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return hypot(v1, v2);
    
  }
  static std::string name() { return "hypot_vv"; }
};

template <typename T>
class hypot_vd_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return hypot(v, 0.4847);
    
  }
  static std::string name() { return "hypot_vd"; }
};

template <typename T>
class hypot_dv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return hypot(0.3898, v);
    
  }
  static std::string name() { return "hypot_dv"; }
};

TEST(ScalarSmoothFunctions, Hypot) {
  Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
  x1[0] *= 0.576;
  x1[1] *= -0.294;
  
  nomad::tests::test_derivatives<true, true, hypot_vv_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  x2 *= 0.576;
  
  nomad::tests::test_derivatives<true, true, hypot_vd_func>(x2);
  nomad::tests::test_derivatives<true, true, hypot_dv_func>(x2);
}

