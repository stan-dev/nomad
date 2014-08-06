#ifndef nomad__scalar__functions__smooth_functions__inv_hpp
#define nomad__scalar__functions__smooth_functions__inv_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double inv(double x) {
    return 1.0 / x;
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    inv(const var<autodiff_order, strict_smoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double val = 1.0 / input.first_val();
    
    push_dual_numbers<autodiff_order>(val);
    
    push_inputs(input.dual_numbers());
    
    double dv = -val * val;
    
    if (autodiff_order >= 1) push_partials(dv);
    if (autodiff_order >= 2) push_partials(dv *= -2 * val);
    if (autodiff_order >= 3) push_partials(dv *= -3 * val);

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
