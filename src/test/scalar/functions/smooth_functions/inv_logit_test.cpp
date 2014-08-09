#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/base_functor.hpp>
#include <src/scalar/functions.hpp>
#include <src/test/finite_difference.hpp>

template <typename T>
class inv_logit_func: public nomad::base_functor<T> {
public:
  T operator()(const Eigen::VectorXd& x) const {
    T v = x[0];
    return inv_logit(v);
    
  }
  static std::string name() { return "inv_logit"; }
};

void test_inv_logit() {

}

TEST(ScalarSmoothFunctions, InvLogit) {
  Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
  x *= 0.576;
  nomad::tests::test_function<true, inv_logit_func>(x);
}
