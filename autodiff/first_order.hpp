#ifndef nomad__autodiff__first_order_hpp
#define nomad__autodiff__first_order_hpp

#include <iomanip>
#include <string>
#include <Eigen/Core>

#include <var/var.hpp>
#include <autodiff/exceptions.hpp>

namespace nomad {

  template<class T_var>
  void first_order_forward_adj(const T_var& v) {
    for (nomad_idx_t i = 1; i <= v.body(); ++i)
      var_bodies_[i].first_order_forward_adj();
  }
  
  template<class T_var>
  void first_order_reverse_adj(const T_var& v) {
    var_bodies_[v.body()].first_grad() = 1.0;
    for (nomad_idx_t i = v.body(); i > 0; --i)
      var_bodies_[i].first_order_reverse_adj();
  }

  template <typename F>
  void gradient(const F& f,
                const Eigen::VectorXd& x,
                double& fx,
                Eigen::VectorXd& g) {
    
    reset();

    try {
      
      auto fx_var = f(x);
      
      if (fx_var.order() < 1)
        throw autodiff_fail_ex("Autodiff order "
                               + std::to_string(fx_var.order())
                               + " insufficient for computing the gradient");
      fx = fx_var.first_val();
      first_order_reverse_adj(fx_var);
      
    } catch (...) {
      reset();
      throw;
    }
    
    for (eigen_idx_t i = 0; i < x.size(); ++i)
      g(i) = var_bodies_[i + 1].first_grad();

    reset();
    
  }
  
  template <typename F>
  void gradient(const F& f,
                const Eigen::VectorXd& x,
                Eigen::VectorXd& g) {
    double fx;
    gradient(f, x, fx, g);
  }
  
  template <typename F>
  void finite_diff_gradient(const F& functional,
                            const Eigen::VectorXd& x,
                            Eigen::VectorXd& g,
                            const double epsilon = 1e-6) {
    
    Eigen::VectorXd x_dynam(x);
    
    try {
      
      auto f_var = functional(x);
      reset();
      
      if (f_var.order() < 0) {
        throw autodiff_fail_ex("Autodiff order " + std::to_string(f_var.order())
                               + " insufficient for computing the finite difference gradient");
      }
      
    } catch (partial_fail_ex& e) {
      std::cout << "Finite difference gradient cannot be computed -- " << std::endl;
      std::cout << e.what() << std::endl;
      reset();
      throw e;
    }
    
    for (eigen_idx_t i = 0; i < x.size(); ++i) {
      
      double delta_f = 0;
      
      x_dynam(i) += epsilon;
      auto v1 = functional(x_dynam);
      delta_f += v1.first_val();
      reset();
      
      x_dynam(i) -= 2.0 * epsilon;
      auto v2 = functional(x_dynam);
      delta_f -= v2.first_val();
      reset();
      
      x_dynam(i) += epsilon;
      
      delta_f /= 2.0 * epsilon;
      
      g(i) = delta_f;
      
    }
    
  }
  
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
    
    std::cout.precision(6);
    int width = 12;
    int n_column = 4;
    
    std::cout << "Gradient Test:" << std::endl;
    std::cout << "    " << std::setw(n_column * width) << std::setfill('-')
              << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Component"
              << std::setw(width) << std::left << "Automatic"
              << std::setw(width) << std::left << "Finite"
              << std::setw(width) << std::left << "Delta / "
              << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "(i)"
              << std::setw(width) << std::left << "Derivative"
              << std::setw(width) << std::left << "Difference"
              << std::setw(width) << std::left << "Stepsize^{2}"
              << std::endl;
    std::cout << "    " << std::setw(n_column * width) << std::setfill('-')
              << "" << std::setfill(' ') << std::endl;
    
    Eigen::VectorXd x_dynam(x);
    
    for (eigen_idx_t i = 0; i < x.size(); ++i) {
      std::cout << "    "
                << std::setw(width) << std::left << i
                << std::setw(width) << std::left << auto_grad(i)
                << std::setw(width) << std::left << diff_grad(i)
                << std::setw(width) << std::left << (auto_grad(i) - diff_grad(i)) / (epsilon * epsilon)
                << std::endl;
      
    }
    
    std::cout << "    " << std::setw(n_column * width) << std::setfill('-')
              << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
  }
  
  template <typename F>
  void gradient_dot_vector(const F& functional,
                           const Eigen::VectorXd& x,
                           const Eigen::VectorXd& v,
                           double& f,
                           double& grad_dot_v) {
    
    reset();
    
    try {
      
      auto f_var = functional(x);
      
      if (f_var.order() < 1) {
        reset();
        throw autodiff_fail_ex("Autodiff order " + std::to_string(f_var.order())
                               + " insufficient for computing the gradient dot vector");
      }
      
      f = f_var.first_val();
    
      for (eigen_idx_t i = 0; i < x.size(); ++i)
        var_bodies_[i + 1].first_grad() = v(i);
      
      first_order_forward_adj(f_var);
      
      grad_dot_v = f_var.first_grad();

    } catch (partial_fail_ex& e) {
      std::cout << "Gradient dot vector cannot be computed -- " << std::endl;
      std::cout << e.what() << std::endl;
      reset();
      throw e;
    }
    
    reset();
    
  }
  
  template <typename F>
  void gradient_dot_vector(const F& functional,
                           const Eigen::VectorXd& x,
                           const Eigen::VectorXd& v,
                           double& grad_dot_v) {
    double f;
    try {
      gradient_dot_vector(functional, x, v, f, grad_dot_v);
    } catch (std::runtime_error& e) {
      throw e;
    }
  }
  
  template <typename F>
  void test_gradient_dot_vector(const F& functional,
                                const Eigen::VectorXd& x,
                                const Eigen::VectorXd& v) {
    
    Eigen::VectorXd g_auto(x.size());
    try {
      gradient(functional, x, g_auto);
    } catch (std::runtime_error& e) {
      std::cout << "Cannot compute Gradient Dot Vector Test" << std::endl;
      std::cout << e.what() << std::endl;
    }
    
    double g_dot_v = g_auto.dot(v);
    
    double g_dot_v_auto;
    try {
      gradient_dot_vector(functional, x, v, g_dot_v_auto);
    } catch (std::runtime_error& e) {
      std::cout << "Cannot compute Gradient Dot Vector Test" << std::endl;
      std::cout << e.what() << std::endl;
    }
    
    std::cout.precision(6);
    int width = 12;
    int n_column = 2;
    
    std::cout << "Gradient Dot Vector Test:" << std::endl;
    std::cout << "    " << std::setw(n_column * width) << std::setfill('-')
              << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Automatic"
              << std::setw(width) << std::left << "Exact"
              << std::endl;
    std::cout << "    "
              << std::setw(width) << std::left << "Derivative"
              << std::setw(width) << std::left << ""
              << std::endl;
    std::cout << "    " << std::setw(n_column * width) << std::setfill('-')
              << "" << std::setfill(' ') << std::endl;
    
    std::cout << "    "
              << std::setw(width) << std::left << g_dot_v_auto
              << std::setw(width) << std::left << g_dot_v
              << std::endl;
    
    std::cout << "    " << std::setw(n_column * width) << std::setfill('-')
              << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
  }
  
}

#endif
