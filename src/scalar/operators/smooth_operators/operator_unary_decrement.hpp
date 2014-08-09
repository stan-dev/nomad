#ifndef nomad__src__scalar__operators__smooth_operators__operator_unary_decrement_hpp
#define nomad__src__scalar__operators__smooth_operators__operator_unary_decrement_hpp

#include <src/var/var.hpp>
#include <src/var/derived/unary_plus_var_body.hpp>

namespace nomad {

  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>&
    operator--(var<autodiff_order, strict_smoothness>& v1) {
    
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    // next_partials_delta not used by unary_plus_var_body
    
    new unary_plus_var_body<autodiff_order>();
    
    push_dual_numbers<autodiff_order>(v1.first_val() - 1.0);
    
    push_inputs(v1.dual_numbers());

    v1.set_body(next_body_idx_ - 1);
    return v1;
    
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    operator--(const var<autodiff_order, strict_smoothness>& v1, int /* dummy */) {

    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    // next_partials_delta not used by unary_plus_var_body
    
    new unary_plus_var_body<autodiff_order>();
    
    push_dual_numbers<autodiff_order>(v1.first_val() - 1.0);
    
    push_inputs(v1.dual_numbers());
    
    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
