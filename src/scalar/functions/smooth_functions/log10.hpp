#ifndef nomad__src__scalar__functions__smooth_functions__log10_hpp
#define nomad__src__scalar__functions__smooth_functions__log10_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double log10(double x) { return std::log10(x); }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    log10(const var<autodiff_order, strict_smoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double val = input.first_val();
    
    push_dual_numbers<autodiff_order>(log10(val));
    
    push_inputs(input.dual_numbers());
    
    double val_inv = 1.0 / val;
    
    if (autodiff_order >= 1) push_partials(val = val_inv * 4.48142011772455);
    if (autodiff_order >= 2) push_partials(val *= - val_inv);
    if (autodiff_order >= 3) push_partials(val *= - 2.0 * val_inv);

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
