#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class pow_vv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return pow(nomad::tests::construct_unsafe_var<T>(x[0]),
               nomad::tests::construct_unsafe_var<T>(x[1]));
    
  }
  static std::string name() { return "pow_vv"; }
};

template <typename T>
class pow_vd_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return pow(nomad::tests::construct_unsafe_var<T>(x[0]),
               x[1]);
    
  }
  static std::string name() { return "pow_vd"; }
};

template <typename T>
class pow_dv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return pow(x[0],
               nomad::tests::construct_unsafe_var<T>(x[1]));
    
  }
  static std::string name() { return "pow_dv"; }
};

template <typename T>
class pow_vv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return pow(T(x[0]), T(x[1]));
    
  }
  static std::string name() { return "pow_vv"; }
};

template <typename T>
class pow_vd_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return pow(T(x[0]), 0.4847);
    
  }
  static std::string name() { return "pow_vd"; }
};

template <typename T>
class pow_dv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return pow(0.3898, T(x[0]));
    
  }
  static std::string name() { return "pow_dv"; }
};

TEST(ScalarSmoothFunctions, Pow) {

  nomad::eigen_idx_t d = 2;
  
  Eigen::VectorXd x1(d);
  x1[0] = 0.576;
  x1[1] = -0.294;
  
  Eigen::MatrixXd x1_bad(d, 1);
  x1_bad(0, 0) = -0.294;
  x1_bad(1, 0) = 0.576;
  
  nomad::tests::test_validation<pow_vv_eval_func>(x1, x1_bad);
  nomad::tests::test_validation<pow_vd_eval_func>(x1, x1_bad);
  nomad::tests::test_validation<pow_dv_eval_func>(x1, x1_bad);
  
  nomad::tests::test_derivatives<pow_vv_grad_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  x2 *= 0.576;
  
  nomad::tests::test_derivatives<pow_vd_grad_func>(x2);
  nomad::tests::test_derivatives<pow_dv_grad_func>(x2);
}

