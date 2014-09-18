#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/io_validation.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class trinary_prod_cubes_eval_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    return trinary_prod_cubes(nomad::tests::construct_unsafe_var<T>(x[0]),
                              nomad::tests::construct_unsafe_var<T>(x[1]),
                              nomad::tests::construct_unsafe_var<T>(x[2]));
    
  }
  static std::string name() { return "trinary_prod_cubes"; }
};

template <typename T>
class trinary_prod_cubes_grad_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    T v3 = x[2];
    return trinary_prod_cubes(v1, v2, v3);
    
  }
  static std::string name() { return "trinary_prod_cubes"; }
};

TEST(ScalarSmoothFunctions, TrinaryProdCubes) {
  
  nomad::eigen_idx_t d = 3;
  
  Eigen::VectorXd x(d);
  x[0] = 0.576;
  x[1] = 0.832;
  x[2] = -1.765;
  
  nomad::tests::test_validation<trinary_prod_cubes_eval_func>(x);
  nomad::tests::test_derivatives<trinary_prod_cubes_grad_func>(x);
}

