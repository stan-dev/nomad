#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class binary_prod_cubes_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v1 = x[0];
    T v2 = x[1];
    return binary_prod_cubes(v1, v2);
    
  }
  static std::string name() { return "binary_prod_cubes"; }
};

TEST(ScalarSmoothFunctions, BinaryProdCubes) {
  Eigen::VectorXd x = Eigen::VectorXd::Ones(2);
  x[0] = 0.576;
  x[1] = 0.832;
  nomad::tests::test_derivatives<true, true, binary_prod_cubes_func>(x);
}

