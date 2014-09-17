#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class hypot_vv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return hypot(nomad::tests::construct_unsafe_var<T>(x[0]),
                 nomad::tests::construct_unsafe_var<T>(x[1]));
    
  }
  static std::string name() { return "hypot_vv"; }
};

template <typename T>
class hypot_vd_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return hypot(nomad::tests::construct_unsafe_var<T>(x[0]),
                 x[1]);
    
  }
  static std::string name() { return "hypot_vd"; }
};

template <typename T>
class hypot_dv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return hypot(x[0],
                 nomad::tests::construct_unsafe_var<T>(x[1]));
    
  }
  static std::string name() { return "hypot_dv"; }
};

template <typename T>
class hypot_vv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return hypot(v1, v2);
    
  }
  static std::string name() { return "hypot_vv"; }
};

template <typename T>
class hypot_vd_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return hypot(T(x[0]), 0.4847);
    
  }
  static std::string name() { return "hypot_vd"; }
};

template <typename T>
class hypot_dv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return hypot(0.3898, T(x[0]));
    
  }
  static std::string name() { return "hypot_dv"; }
};

TEST(ScalarSmoothFunctions, Hypot) {
  Eigen::VectorXd x1 = Eigen::VectorXd::Ones(2);
  x1[0] *= 0.576;
  x1[1] *= -0.294;
  
  nomad::tests::test_validation<hypot_vv_eval_func>(x1);
  nomad::tests::test_validation<hypot_vd_eval_func>(x1);
  nomad::tests::test_validation<hypot_dv_eval_func>(x1);
  
  nomad::tests::test_derivatives<hypot_vv_grad_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  x2 *= 0.576;
  
  nomad::tests::test_derivatives<hypot_vd_grad_func>(x2);
  nomad::tests::test_derivatives<hypot_dv_grad_func>(x2);
}

