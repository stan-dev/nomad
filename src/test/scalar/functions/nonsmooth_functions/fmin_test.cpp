#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class fmin_vv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return fmin(exp(v1), exp(v2));
  }
  static std::string name() { return "fmin_vv"; }
};

template <typename T>
class fmin_vd_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return fmin(exp(v), exp(0.5));
  }
  static std::string name() { return "fmin_vd"; }
};

template <typename T>
class fmin_dv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return fmin(exp(0.5), exp(v));
  }
  static std::string name() { return "fmin_dv"; }
};


TEST(ScalarNonSmoothFunctions, Fmin) {
  Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
  x1[0] = 0.75;
  x1[1] = 0.25;
  
  nomad::tests::test_derivatives<false, false, fmin_vv_func>(x1);
  
  x1 *= -1;
  nomad::tests::test_derivatives<false, false, fmin_vv_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  
  x2[0] = 0.75;
  nomad::tests::test_derivatives<false, false, fmin_vd_func>(x2);
  nomad::tests::test_derivatives<false, false, fmin_dv_func>(x2);
  
  x2[0] = 0.25;
  nomad::tests::test_derivatives<false, false, fmin_vd_func>(x2);
  nomad::tests::test_derivatives<false, false, fmin_dv_func>(x2);
}
