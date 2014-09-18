#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/operators.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class operator_subtraction_assignment_vv_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = nomad::tests::construct_unsafe_var<T>(x[0]);
    T v2 = nomad::tests::construct_unsafe_var<T>(x[1]);
    return v1 -= v2;
  }
  static std::string name() { return "operator_subtraction_assignment_vv"; }
};

template <typename T>
class operator_subtraction_assignment_vd_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return exp(nomad::tests::construct_unsafe_var<T>(x[0]) -= x[1]);
    
  }
  static std::string name() { return "operator_subtraction_assignment_vd"; }
};

template <typename T>
class operator_subtraction_assignment_vv_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return exp(v1 -= v2);
  }
  static std::string name() { return "operator_subtraction_assignment_vv"; }
};

template <typename T>
class operator_subtraction_assignment_vd_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return exp(v -= 0.4847);
    
  }
  static std::string name() { return "operator_subtraction_assignment_vd"; }
};

TEST(ScalarSmoothOperators, OperatorSubtractionAssignment) {
  nomad::eigen_idx_t d = 2;
  
  Eigen::VectorXd x1(d);
  x1[0] = 0.576;
  x1[1] = -0.294;

  nomad::tests::test_validation<operator_subtraction_assignment_vv_eval_func>(x1);
  nomad::tests::test_validation<operator_subtraction_assignment_vv_eval_func>(x1);
  
  nomad::tests::test_derivatives<operator_subtraction_assignment_vv_grad_func>(x1);
  
  Eigen::VectorXd x2 = Eigen::VectorXd::Ones(1);
  x2 *= 0.576;
  
  nomad::tests::test_derivatives<operator_subtraction_assignment_vd_grad_func>(x2);
}

