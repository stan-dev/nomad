#ifndef nomad__tests__validate_exceptions_hpp
#define nomad__tests__validate_exceptions_hpp

#include <tests/var_fail.hpp>
#include <autodiff/exceptions.hpp>

namespace nomad {

  template <typename T>
  struct f_first_fail {
    T operator()(const Eigen::VectorXd& x) const {
      first_fail<T>();
      return T();
    }
  };

  template <typename T>
  struct f_second_fail {
    T operator()(const Eigen::VectorXd& x) const {
      second_fail<T>();
      return T();
    }
  };

  template <typename T>
  struct f_third_fail {
    T operator()(const Eigen::VectorXd& x) const {
      third_fail<T>();
      return T();
    }
  };

  void validate_exceptions() {
    Eigen::VectorXd y = Eigen::VectorXd::Zero(10);
    
    std::cout << "First Fail: " << std::endl;
    
    f_first_fail<var<0U> > first_zero;
    try {
      first_zero(y);
      std::cout << "Success!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << e.what() << std::endl;
    }
    
    f_first_fail<var<1U> > first_first;
    try {
      first_first(y);
      std::cout << "Success!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << e.what() << std::endl;
    }
    
    f_first_fail<var<2U> > first_second;
    try {
      first_second(y);
      std::cout << "Success!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << e.what() << std::endl;
    }
    
    f_first_fail<var<3U> > first_third;
    try {
      first_third(y);
      std::cout << "Success!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << e.what() << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Second Fail: " << std::endl;
    
    f_second_fail<var<0U> > second_zero;
    try {
      second_zero(y);
      std::cout << "Success!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << e.what() << std::endl;
    }
    
    f_second_fail<var<1U> > second_first;
    try {
      second_first(y);
      std::cout << "Success!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << e.what() << std::endl;
    }
    
    f_second_fail<var<2U> > second_second;
    try {
      second_second(y);
      std::cout << "Success!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << e.what() << std::endl;
    }
    
    f_second_fail<var<3U> > second_third;
    try {
      second_third(y);
      std::cout << "Success!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << e.what() << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Third Fail: " << std::endl;
    
    f_third_fail<var<0U> > third_zero;
    try {
      third_zero(y);
      std::cout << "Success!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << e.what() << std::endl;
    }
    
    f_third_fail<var<1U> > third_first;
    try {
      third_first(y);
      std::cout << "Success!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << e.what() << std::endl;
    }
    
    f_third_fail<var<2U> > third_second;
    try {
      third_second(y);
      std::cout << "Success!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << e.what() << std::endl;
    }
    
    f_third_fail<var<3U> > third_third;
    try {
      third_third(y);
      std::cout << "Success!" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << e.what() << std::endl;
    }
    
    /*
     // Gradient testing
     Eigen::VectorXd x = Eigen::VectorXd::Ones(101);
     Eigen::VectorXd v = Eigen::VectorXd::Ones(101);
     Eigen::MatrixXd M = Eigen::MatrixXd::Identity(101, 101);
     
     double f;
     Eigen::VectorXd grad(x.size());
     Eigen::MatrixXd H(x.size(), x.size());
     Eigen::MatrixXd grad_H(x.size(), x.size() * x.size());
     Eigen::VectorXd grad_m_times_h(x.size());
     
     funnel_func<var<0U> > zero_order_funnel;
     funnel_func<var<1U> > first_order_funnel;
     funnel_func<var<2U> > second_order_funnel;
     funnel_func<var<3U> > third_order_funnel;
     
     std::cout << std::endl << "Gradient:" << std::endl;
     
     try {
     gradient(zero_order_funnel, x, f, grad);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     gradient(first_order_funnel, x, f, grad);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     gradient(second_order_funnel, x, f, grad);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     gradient(third_order_funnel, x, f, grad);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     std::cout << std::endl << "Finite Difference Gradient:" << std::endl;
     
     try {
     finite_diff_gradient(zero_order_funnel, x, grad);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     finite_diff_gradient(first_order_funnel, x, grad);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     finite_diff_gradient(second_order_funnel, x, grad);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     finite_diff_gradient(third_order_funnel, x, grad);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     */
    /*
     std::cout << std::endl << "Hessian:" << std::endl;
     
     try {
     hessian(zero_order_funnel, x, f, grad, H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     hessian(first_order_funnel, x, f, grad, H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     hessian(second_order_funnel, x, f, grad, H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     hessian(third_order_funnel, x, f, grad, H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     std::cout << std::endl << "Finite Difference Hessian:" << std::endl;
     
     try {
     finite_diff_hessian(zero_order_funnel, x, H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     finite_diff_hessian(first_order_funnel, x, H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     finite_diff_hessian(second_order_funnel, x, H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     finite_diff_hessian(third_order_funnel, x, H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     std::cout << std::endl << "Grad Hessian:" << std::endl;
     
     try {
     grad_hessian(zero_order_funnel, x, f, grad, H, grad_H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     grad_hessian(first_order_funnel, x, f, grad, H, grad_H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     grad_hessian(second_order_funnel, x, f, grad, H, grad_H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     grad_hessian(third_order_funnel, x, f, grad, H, grad_H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     std::cout << std::endl << "Finite Difference Grad Hessian:" << std::endl;
     
     try {
     finite_diff_grad_hessian(zero_order_funnel, x, grad_H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     finite_diff_grad_hessian(first_order_funnel, x, grad_H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     finite_diff_grad_hessian(second_order_funnel, x, grad_H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     
     try {
     finite_diff_grad_hessian(third_order_funnel, x, grad_H);
     std::cout << "Success!" << std::endl;
     } catch (std::runtime_error& e) {
     std::cout << e.what() << std::endl;
     }
     */
  }
  
}
  
#endif
