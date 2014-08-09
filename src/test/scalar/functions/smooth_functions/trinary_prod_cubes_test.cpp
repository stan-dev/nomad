#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class trinary_prod_cubes_func: public nomad::base_functor<T> {
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
  Eigen::VectorXd x = Eigen::VectorXd::Ones(3);
  x[0] = 0.576;
  x[1] = 0.832;
  x[1] = -1.765;
  nomad::tests::test_function<true, trinary_prod_cubes_func>(x);
}

