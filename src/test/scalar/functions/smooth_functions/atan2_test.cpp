#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class atan2_vv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return atan2(nomad::tests::construct_unsafe_var<T>(x[0]),
                 nomad::tests::construct_unsafe_var<T>(x[1]));
  }
  static std::string name() { return "atan2_vv"; }
};

template <typename T>
class atan2_vd_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return atan2(nomad::tests::construct_unsafe_var<T>(x[0]),
                 x[1]);
  }
  static std::string name() { return "atan2_vd"; }
};

template <typename T>
class atan2_dv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return atan2(x[0],
                 nomad::tests::construct_unsafe_var<T>(x[1]));
  }
  static std::string name() { return "atan2_dv"; }
};

template <typename T>
class atan2_vv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return atan2(T(x[0]), T(x[1]));
    
  }
  static std::string name() { return "atan2_vv"; }
};

template <typename T>
class atan2_vd_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return atan2(T(x[0]), 0.4847);
    
  }
  static std::string name() { return "atan2_vd"; }
};

template <typename T>
class atan2_dv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return atan2(0.3898, T(x[0]));
    
  }
  static std::string name() { return "atan2_dv"; }
};


TEST(ScalarSmoothFunctions, Atan2) {
  
  nomad::eigen_idx_t d = 2;
  
  Eigen::VectorXd x1(d);
  x1[0] = 0.576;
  x1[1] = -0.294;
  
  nomad::tests::test_validation<atan2_vv_eval_func>(x1);
  nomad::tests::test_validation<atan2_vd_eval_func>(x1);
  nomad::tests::test_validation<atan2_dv_eval_func>(x1);
  
  nomad::tests::test_derivatives<atan2_vv_grad_func>(x1);
  
  Eigen::VectorXd x2(1);
  x2[0] = 0.576;
  
  nomad::tests::test_derivatives<atan2_vd_grad_func>(x2);
  nomad::tests::test_derivatives<atan2_dv_grad_func>(x2);
}
