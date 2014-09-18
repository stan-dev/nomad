#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class fmod_vv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fmod(nomad::tests::construct_unsafe_var<T>(x[0]),
                nomad::tests::construct_unsafe_var<T>(x[1]));
  }
  static std::string name() { return "fmod_vv"; }
};

template <typename T>
class fmod_vd_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fmod(nomad::tests::construct_unsafe_var<T>(x[0]),
                x[1]);
  }
  static std::string name() { return "fmod_vd"; }
};

template <typename T>
class fmod_dv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return fmod(x[0],
                nomad::tests::construct_unsafe_var<T>(x[1]));
  }
  static std::string name() { return "fmod_dv"; }
};

template <typename T>
class fmod_vv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return exp(fmod(T(x[0]), T(x[1])));
  }
  static std::string name() { return "fmod_vv"; }
};

template <typename T>
class fmod_vd_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return exp(fmod(T(x[0]), 3.0));
  }
  static std::string name() { return "fmod_vd"; }
};

template <typename T>
class fmod_dv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return exp(fmod(3.25, T(x[0])));
  }
  static std::string name() { return "fmod_dv"; }
};

TEST(ScalarNonSmoothFunctions, Fmax) {

  nomad::eigen_idx_t d = 2;
  
  Eigen::VectorXd x1(d);
  x1[0] = 3.25;
  x1[1] = 3.0;
  
  nomad::tests::test_validation<fmod_vv_eval_func>(x1);
  nomad::tests::test_validation<fmod_vd_eval_func>(x1);
  nomad::tests::test_validation<fmod_dv_eval_func>(x1);
  
  nomad::tests::test_derivatives<fmod_vv_grad_func>(x1);

  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  
  x2[0] = 3.25;
  nomad::tests::test_derivatives<fmod_vd_grad_func>(x2);
  
  x2[0] = 3.0;
  nomad::tests::test_derivatives<fmod_dv_grad_func>(x2);
}
