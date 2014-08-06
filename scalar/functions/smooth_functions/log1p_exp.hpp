#ifndef nomad__scalar__functions__smooth_functions__log1p_exp_hpp
#define nomad__scalar__functions__smooth_functions__log1p_exp_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double log1p_exp(double x) {
    if (x > 0) return x + log1p(exp(-x));
    return std::log1p(exp(x));
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    log1p_exp(const var<autodiff_order, strict_smoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double val = input.first_val();
    
    push_dual_numbers<autodiff_order>(log1p_exp(val));
    
    push_inputs(input.dual_numbers());
    
    double val_inv = 1.0 / (1 + val);
    
    if (val > 0) {
      double e = exp(-val);
      double p = 1.0 / (1.0 + e);
      if (autodiff_order >= 1) push_partials(p);
      if (autodiff_order >= 2) push_partials(p * p * e);
      if (autodiff_order >= 3) push_partials(p * (2.0 * p * p - 3.0 * p + 1.0));
    } else {
      double e = exp(val);
      double p = e / (1.0 + e);
      if (autodiff_order >= 1) push_partials(p);
      if (autodiff_order >= 2) push_partials(p * p / e);
      if (autodiff_order >= 3) push_partials(p * (2.0 * p * p - 3.0 * p + 1.0));
    }

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
