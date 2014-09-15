#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class fdim_vv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return exp(fdim(v1, v2));
  }
  static std::string name() { return "fdim_vv"; }
};

template <typename T>
class fdim_vd_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return exp(fdim(v, 0.5));
  }
  static std::string name() { return "fdim_vd"; }
};

template <typename T>
class fdim_dv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return exp(fdim(1.0, v));
  }
  static std::string name() { return "fdim_dv"; }
};


TEST(ScalarNonSmoothFunctions, Fdim) {
  Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
  x1[0] *= 0.5;
  x1[1] *= -0.5;
  
  nomad::tests::test_function<false, false, fdim_vv_func>(x1);
  
  x1 *= -1;
  nomad::tests::test_function<false, false, fdim_vv_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  nomad::tests::test_function<false, false, fdim_vd_func>(x2);
  
  x2[0] = 0.5;
  nomad::tests::test_function<false, false, fdim_dv_func>(x2);

  x2[0] = -1.0;
  nomad::tests::test_function<false, false, fdim_vd_func>(x2);
  nomad::tests::test_function<false, false, fdim_dv_func>(x2);
}
