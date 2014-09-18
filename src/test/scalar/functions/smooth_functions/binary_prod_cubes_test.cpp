#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class binary_prod_cubes_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return binary_prod_cubes(nomad::tests::construct_unsafe_var<T>(x[0]),
                             nomad::tests::construct_unsafe_var<T>(x[1]));
    
  }
  static std::string name() { return "binary_prod_cubes"; }
};

template <typename T>
class binary_prod_cubes_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return binary_prod_cubes(T(x[0]), T(x[1]));
    
  }
  static std::string name() { return "binary_prod_cubes"; }
};

TEST(ScalarSmoothFunctions, BinaryProdCubes) {
  
  nomad::eigen_idx_t d = 2;
  Eigen::VectorXd x(d);
  x[0] = 0.576;
  x[1] = 0.832;
  
  nomad::tests::test_validation<binary_prod_cubes_eval_func>(x);
  nomad::tests::test_derivatives<binary_prod_cubes_grad_func>(x);
}

