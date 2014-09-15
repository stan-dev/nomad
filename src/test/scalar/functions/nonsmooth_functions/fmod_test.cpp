#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class fmod_vv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return exp(fmod(v1, v2));
  }
  static std::string name() { return "fmod_vv"; }
};

template <typename T>
class fmod_vd_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return exp(fmod(v, 3.0));
  }
  static std::string name() { return "fmod_vd"; }
};

template <typename T>
class fmod_dv_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return exp(fmod(3.25, v));
  }
  static std::string name() { return "fmod_dv"; }
};


TEST(ScalarNonSmoothFunctions, Fmax) {
  Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
  x1[0] = 3.25;
  x1[1] = 3.0;
  nomad::tests::test_function<false, false, fmod_vv_func>(x1);

  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  
  x2[0] = 3.25;
  nomad::tests::test_function<false, false, fmod_vd_func>(x2);
  
  x2[0] = 3.0;
  nomad::tests::test_function<false, false, fmod_dv_func>(x2);
}
