#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class fmin_vv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fmin(nomad::tests::construct_unsafe_var<T>(x[0]),
                nomad::tests::construct_unsafe_var<T>(x[1]));
  }
  static std::string name() { return "fmin_vv"; }
};

template <typename T>
class fmin_vd_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fmin(nomad::tests::construct_unsafe_var<T>(x[0]),
                x[1]);
  }
  static std::string name() { return "fmin_vd"; }
};

template <typename T>
class fmin_dv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fmin(x[0],
                nomad::tests::construct_unsafe_var<T>(x[1]));
  }
  static std::string name() { return "fmin_dv"; }
};

template <typename T>
class fmin_vv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return fmin(exp(v1), exp(v2));
  }
  static std::string name() { return "fmin_vv"; }
};

template <typename T>
class fmin_vd_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fmin(exp(T(x[0])), exp(0.5));
  }
  static std::string name() { return "fmin_vd"; }
};

template <typename T>
class fmin_dv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fmin(exp(0.5), exp(T(x[0])));
  }
  static std::string name() { return "fmin_dv"; }
};

TEST(ScalarNonSmoothFunctions, Fmin) {

  nomad::eigen_idx_t d = 2;
  
  Eigen::VectorXd x1(d);
  x1[0] = 0.75;
  x1[1] = 0.25;

  nomad::tests::test_validation<fmin_vv_eval_func>(x1);
  nomad::tests::test_validation<fmin_vd_eval_func>(x1);
  nomad::tests::test_validation<fmin_dv_eval_func>(x1);
  
  nomad::tests::test_derivatives<fmin_vv_grad_func>(x1);
  
  x1 *= -1;
  nomad::tests::test_derivatives<fmin_vv_grad_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  
  x2[0] = 0.75;
  nomad::tests::test_derivatives<fmin_vd_grad_func>(x2);
  nomad::tests::test_derivatives<fmin_dv_grad_func>(x2);
  
  x2[0] = 0.25;
  nomad::tests::test_derivatives<fmin_vd_grad_func>(x2);
  nomad::tests::test_derivatives<fmin_dv_grad_func>(x2);
}
