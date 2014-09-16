#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class fmax_vv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return fmax(exp(v1), exp(v2));
  }
  static std::string name() { return "fmax_vv"; }
};

template <typename T>
class fmax_vd_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return fmax(exp(v), exp(0.5));
  }
  static std::string name() { return "fmax_vd"; }
};

template <typename T>
class fmax_dv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return fmax(exp(0.5), exp(v));
  }
  static std::string name() { return "fmax_dv"; }
};


TEST(ScalarNonSmoothFunctions, Fmax) {
  Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
  x1[0] = 0.75;
  x1[1] = 0.25;
  
  nomad::tests::test_derivatives<false, false, fmax_vv_func>(x1);
  
  x1 *= -1;
  nomad::tests::test_derivatives<false, false, fmax_vv_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  
  x2[0] = 0.75;
  nomad::tests::test_derivatives<false, false, fmax_vd_func>(x2);
  nomad::tests::test_derivatives<false, false, fmax_dv_func>(x2);
  
  x2[0] = 0.25;
  nomad::tests::test_derivatives<false, false, fmax_vd_func>(x2);
  nomad::tests::test_derivatives<false, false, fmax_dv_func>(x2);
}
