#ifndef nomad__scalar__functions__smooth_functions__log_sum_exp_hpp
#define nomad__scalar__functions__smooth_functions__log_sum_exp_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>
#include <var/derived/binary_var_body.hpp>

namespace nomad {
  
  inline double log_sum_exp(double x, double y) {
    if (x > y)
      return x + std::log(std::exp(y - x) + 1);
    else
      return y + std::log(std::exp(x - y) + 1);
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    log_sum_exp(const var<autodiff_order, strict_smoothness>& v1,
                const var<autodiff_order, strict_smoothness>& v2) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 2;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      binary_var_body<autodiff_order, partials_order>::n_partials();
    
    new binary_var_body<autodiff_order, partials_order>();

    double x = v1.first_val();
    double y = v2.first_val();
    
    push_dual_numbers<autodiff_order>(log_sum_exp(x, y));
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    if (x > y) {
    
      double e = std::exp(y - x);
      double p = e / (1.0 + e);
      
      if (autodiff_order >= 1) {
        push_partials(1.0 / (1.0 + e));
        push_partials(p);
      }
      if (autodiff_order >= 2) {
        double p2 = p * p / e;
        push_partials(p2);
        push_partials(-p2);
        push_partials(p2);
      }
      if (autodiff_order >= 3) {
        p *= 2 * p * p - 3 * p + 1;
        push_partials(-p);
        push_partials(p);
        push_partials(-p);
        push_partials(p);
      }
      
    } else {
      
      double e = std::exp(x - y);
      double p = e / (1.0 + e);
      
      if (autodiff_order >= 1) {
        push_partials(p);
        push_partials(1.0 / (1.0 + e));
      }
      if (autodiff_order >= 2) {
        double p2 = p * p / e;
        push_partials(p2);
        push_partials(-p2);
        push_partials(p2);
      }
      if (autodiff_order >= 3) {
        p *= 2 * p * p - 3 * p + 1;
        push_partials(p);
        push_partials(-p);
        push_partials(p);
        push_partials(-p);
      }
      
    }

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    log_sum_exp(double x,
                const var<autodiff_order, strict_smoothness>& v2) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();
    
    double y = v2.first_val();
    push_dual_numbers<autodiff_order>(log_sum_exp(x, y));
    
    push_inputs(v2.dual_numbers());
    
    if (x > y) {
      
      double e = std::exp(y - x);
      double p = e / (1.0 + e);
      
      if (autodiff_order >= 1) {
        push_partials(p);
      }
      if (autodiff_order >= 2) {
        push_partials(p * p / e);
      }
      if (autodiff_order >= 3) {
        p *= 2 * p * p - 3 * p + 1;
        push_partials(p);
      }
      
    } else {
      
      double e = std::exp(x - y);
      double p = 1.0 / (1.0 + e);
      
      if (autodiff_order >= 1) {
        push_partials(p);
      }
      if (autodiff_order >= 2) {
        p *= e;
        push_partials(p * p / e);
      }
      if (autodiff_order >= 3) {
        p *= 2 * p * p - 3 * p + 1;
        push_partials(-p);
      }
      
    }
    
    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    log_sum_exp(const var<autodiff_order, strict_smoothness>& v1,
                double y) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();
    
    double x = v1.first_val();
    push_dual_numbers<autodiff_order>(log_sum_exp(x, y));
    
    push_inputs(v1.dual_numbers());
    
    if (x > y) {
      
      double e = std::exp(y - x);
      double p = 1.0 / (1.0 + e);
      
      if (autodiff_order >= 1) {
        push_partials(p);
      }
      if (autodiff_order >= 2) {
        p *= e;
        push_partials(p * p / e);
      }
      if (autodiff_order >= 3) {
        p *= 2 * p * p - 3 * p + 1;
        push_partials(-p);
      }
      
    } else {
      
      double e = std::exp(x - y);
      double p = e / (1.0 + e);
      
      if (autodiff_order >= 1) {
        push_partials(p);
      }
      if (autodiff_order >= 2) {
        push_partials(p * p / e);
      }
      if (autodiff_order >= 3) {
        p *= 2 * p * p - 3 * p + 1;
        push_partials(p);
      }
      
    }
    
    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif