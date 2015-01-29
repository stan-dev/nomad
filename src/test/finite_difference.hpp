#ifndef nomad__src__test__finite_difference_hpp
#define nomad__src__test__finite_difference_hpp

#include <gtest/gtest.h>

#include <math.h>
#include <string>

#include <src/autodiff/first_order.hpp>
#include <src/autodiff/second_order.hpp>
#include <src/autodiff/third_order.hpp>

namespace nomad {
  namespace tests {

    template <typename F>
    void test_gradient(const F& functional,
                       const Eigen::VectorXd& x,
                       const double epsilon = 1e-6) {
      
      Eigen::VectorXd auto_grad(x.size());
      try {
        gradient(functional, x, auto_grad);
      } catch (std::runtime_error& e) {
        std::cout << "Cannot compute Gradient Test" << std::endl;
        throw e;
      }
      
      Eigen::VectorXd diff_grad(x.size());
      try {
        finite_diff_gradient(functional, x, diff_grad, epsilon);
      } catch (std::runtime_error& e) {
        std::cout << "Cannot compute Gradient Test" << std::endl;
        throw e;
      }
  
      for (eigen_idx_t i = 0; i < x.size(); ++i) {
        SCOPED_TRACE("\nGradient Finite Difference Test: " + functional.name()
                     + "\n  element " + std::to_string(i)
                     + "\n  auto_diff = " + std::to_string(auto_grad(i))
                     + "\n  finite_diff = " + std::to_string(diff_grad(i)));
        EXPECT_LT(std::fabs(auto_grad(i) - diff_grad(i)), epsilon);
      }
      
    }
  
    template <typename F>
    void test_hessian(const F& functional,
                      const Eigen::VectorXd& x,
                      const double epsilon = 1e-6) {
      
      eigen_idx_t d = x.size();
      
      Eigen::MatrixXd auto_H(x.size(), x.size());
      try {
        hessian(functional, x, auto_H);
      } catch (std::runtime_error& e) {
        std::cout << "Cannot compute Hessian Test" << std::endl;
        throw e;
      }
      
      Eigen::MatrixXd diff_H(x.size(), x.size());
      try {
        finite_diff_hessian(functional, x, diff_H, epsilon);
      } catch (std::runtime_error& e) {
        std::cout << "Cannot compute Hessian Test" << std::endl;
        throw e;
      }

      for (eigen_idx_t i = 0; i < d; ++i) {
        for (eigen_idx_t j = 0; j <= i; ++j) {
          SCOPED_TRACE("Hessian Finite Difference Test: " + functional.name()
                       + "\n  element " + std::to_string(i)
                       + ", " + std::to_string(j)
                       + "\n  auto_diff = " + std::to_string(auto_H(i, j))
                       + "\n  finite_diff = " + std::to_string(diff_H(i, j)));
          EXPECT_LT(std::fabs(auto_H(i, j) - diff_H(i, j)), epsilon);
        }
      }
    }
    
    template <typename F>
    void test_grad_hessian(const F& functional,
                           const Eigen::VectorXd& x,
                           const double epsilon = 1e-6) {
      
      eigen_idx_t d = x.size();
      
      Eigen::MatrixXd auto_grad_H(d, d * d);
      try {
        grad_hessian(functional, x, auto_grad_H);
      } catch (std::runtime_error& e) {
        std::cout << "Cannot compute Hessian Gradient Test" << std::endl;
        throw e;
      }
      
      Eigen::MatrixXd diff_grad_H(d, d * d);
      try {
        finite_diff_grad_hessian(functional, x, diff_grad_H, epsilon);
      } catch (std::runtime_error& e) {
        std::cout << "Cannot compute Hessian Gradient Test" << std::endl;
        throw e;
      }
      
      for (eigen_idx_t k = 0; k < d; ++k) {
        for (eigen_idx_t i = 0; i <= k; ++i) {
          for (eigen_idx_t j = 0; j <= i; ++j) {
            SCOPED_TRACE("Grad Hessian Finite Difference Test: " + functional.name()
                         + "\n  element " + std::to_string(i)
                         + ", " + std::to_string(j)
                         + ", " + std::to_string(k)
                         + "\n  auto_diff = " + std::to_string(auto_grad_H.block(0, k * d, d, d)(i, j))
                         + "\n  finite_diff = " + std::to_string(diff_grad_H.block(0, k * d, d, d)(i, j)));
            EXPECT_LT(std::fabs(  auto_grad_H.block(0, k * d, d, d)(i, j)
                                - diff_grad_H.block(0, k * d, d, d)(i, j) ), epsilon);
          }
        }
      }

    }
    
    template <template <class> class F>
    void test_derivatives(Eigen::VectorXd& x) {
      
      F<var<1U, false, true> > f1;
      try {
        tests::test_gradient(f1, x);
      } catch (nomad_error) {
        SCOPED_TRACE("test_gradient");
        FAIL();
      }
      
      F<var<2U, false, true> > f2;
      try {
        tests::test_hessian(f2, x);
      } catch (nomad_error) {
        SCOPED_TRACE("test_hessian");
        FAIL();
      }
      
      F<var<3U, false, true> > f3;
      try {
        tests::test_grad_hessian(f3, x);
      } catch (nomad_error) {
        SCOPED_TRACE("test_grad_hessian");
        FAIL();
      }
      
    }
  
  }
}

#endif
