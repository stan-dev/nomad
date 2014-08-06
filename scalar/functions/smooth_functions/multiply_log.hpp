#ifndef nomad__scalar__functions__smooth_functions__multiply_log_hpp
#define nomad__scalar__functions__smooth_functions__multiply_log_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>
#include <var/derived/binary_var_body.hpp>

namespace nomad {
  
  inline double multiply_log(double x, double y) {
    if (x == 0.0 && y == 0.0) return 0.0;
    return x * std::log(y);
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    multiply_log(const var<autodiff_order, strict_smoothness>& v1,
                 const var<autodiff_order, strict_smoothness>& v2) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 2;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      binary_var_body<autodiff_order, partials_order>::n_partials();
    
    new binary_var_body<autodiff_order, partials_order>();
    
    double x = v1.first_val();
    double y = v2.first_val();
    
    push_dual_numbers<autodiff_order>(multiply_log(x, y));
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    double y_inv = 1.0 / y;
    
    if (autodiff_order >= 1) {
      push_partials(std::log(y));
      push_partials(x * y_inv);
    }
    if (autodiff_order >= 2) {
      push_partials(0);
      push_partials(y_inv);
      push_partials(- x * y_inv * y_inv);
    }
    if (autodiff_order >= 3) {
      push_partials(0);
      push_partials(0);
      push_partials(- y_inv * y_inv);
      push_partials(2 * x * y_inv * y_inv * y_inv);
    }
    
    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness> multiply_log(double x,
                                          const var<autodiff_order, strict_smoothness>& v2) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
    unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();
    
    double y = v2.first_val();
    
    push_dual_numbers<autodiff_order>(multiply_log(x, y));
    
    push_inputs(v2.dual_numbers());
    
    double y_inv = 1.0 / y;
    
    if (autodiff_order >= 1) push_partials(x * y_inv);
    if (autodiff_order >= 2) push_partials(- x * y_inv * y_inv);
    if (autodiff_order >= 3) push_partials(2 * x * y_inv * y_inv * y_inv);

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness> multiply_log(const var<autodiff_order, strict_smoothness>& v1,
                                          double y) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
    unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();
    
    double x = v1.first_val();
    
    push_dual_numbers<autodiff_order>(multiply_log(x, y));
    
    push_inputs(v1.dual_numbers());
    
    if (autodiff_order >= 1) push_partials(std::log(y));
    
    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif