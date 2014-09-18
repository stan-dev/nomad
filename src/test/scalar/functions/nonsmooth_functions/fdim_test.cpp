#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class fdim_vv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fdim(nomad::tests::construct_unsafe_var<T>(x[0]),
                nomad::tests::construct_unsafe_var<T>(x[1]));
  }
  static std::string name() { return "fdim_vv"; }
};

template <typename T>
class fdim_vd_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fdim(nomad::tests::construct_unsafe_var<T>(x[0]),
                x[1]);
  }
  static std::string name() { return "fdim_vd"; }
};

template <typename T>
class fdim_dv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fdim(x[0],
                nomad::tests::construct_unsafe_var<T>(x[1]));
  }
  static std::string name() { return "fdim_dv"; }
};

template <typename T>
class fdim_vv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return exp(fdim(T(x[0]), T(x[1])));
  }
  static std::string name() { return "fdim_vv"; }
};

template <typename T>
class fdim_vd_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return exp(fdim(T(x[0]), 0.5));
  }
  static std::string name() { return "fdim_vd"; }
};

template <typename T>
class fdim_dv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return exp(fdim(1.0, T(x[0])));
  }
  static std::string name() { return "fdim_dv"; }
};


TEST(ScalarNonSmoothFunctions, Fdim) {

  nomad::eigen_idx_t d = 2;
  
  Eigen::VectorXd x1(d);
  x1[0] = 0.5;
  x1[1] = -0.5;
  
  nomad::tests::test_validation<fdim_vv_eval_func>(x1);
  nomad::tests::test_validation<fdim_vd_eval_func>(x1);
  nomad::tests::test_validation<fdim_dv_eval_func>(x1);
  
  nomad::tests::test_derivatives<fdim_vv_grad_func>(x1);
  
  x1 *= -1;
  nomad::tests::test_derivatives<fdim_vv_grad_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  nomad::tests::test_derivatives<fdim_vd_grad_func>(x2);
  
  x2[0] = 0.5;
  nomad::tests::test_derivatives<fdim_dv_grad_func>(x2);

  x2[0] = -1.0;
  nomad::tests::test_derivatives<fdim_vd_grad_func>(x2);
  nomad::tests::test_derivatives<fdim_dv_grad_func>(x2);
}
