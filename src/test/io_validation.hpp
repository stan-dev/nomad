#ifndef nomad__src__test__io_validation_hpp
#define nomad__src__test__io_validation_hpp

#include <gtest/gtest.h>

#include <math.h>
#include <string>
#include <src/autodiff/exceptions.hpp>

namespace nomad {
  namespace tests {
    
    template <class V>
    V construct_unsafe_var(double x) {
      create_node<var_node<V::order(), 0>>(0);
      push_dual_numbers<V::order(), false>(x);
      return V(next_node_idx_ - 1);
    }
    
    template <template <class> class F>
    void test_validation(Eigen::VectorXd& x_good) {
    
      F<var<0U, false, true> > f;
      
      reset();
      SCOPED_TRACE(f.name() + ": Valid Input Test");
      EXPECT_NO_THROW(auto f_var = f(x_good));
      
      for (eigen_idx_t i = 0; i < x_good.size(); ++i) {
        reset();
        Eigen::VectorXd x_nan = x_good;
        x_nan[i] = std::numeric_limits<double>::quiet_NaN();
        SCOPED_TRACE(f.name() + ": Invalid Input Test (NaN injected into argument "
                     + std::to_string(i) + ")");
        EXPECT_THROW(f(x_nan), nomad_input_error);
      }
      
      reset();
      
    }
    
    template <template <class> class F>
    void test_validation(Eigen::VectorXd& x_good, Eigen::MatrixXd& x_bad) {
      
      F<var<0U, false, true> > f;
      
      reset();
      SCOPED_TRACE(f.name() + ": Valid Input Test");
      EXPECT_NO_THROW(auto f_var = f(x_good));
      
      for (eigen_idx_t i = 0; i < x_bad.cols(); ++i) {
        reset();
        SCOPED_TRACE(f.name() + ": Invalid Input Test (" + std::to_string(i) + "th domain error)");
        EXPECT_THROW(auto f_var = f(x_bad.col(i)), nomad_domain_error);
      }
      
      for (eigen_idx_t i = 0; i < x_good.size(); ++i) {
        reset();
        Eigen::VectorXd x_nan = x_good;
        x_nan[i] = std::numeric_limits<double>::quiet_NaN();
        SCOPED_TRACE(f.name() + ": Invalid Input Test (NaN injected into argument "
                     + std::to_string(i) + ")");
        EXPECT_THROW(f(x_nan), nomad_input_error);
      }
      
      reset();
      
    }
  
  }
}

#endif
