#ifndef nomad__src__scalar__operators__smooth_operators__operator_multiplication_hpp
#define nomad__src__scalar__operators__smooth_operators__operator_multiplication_hpp

#include <src/var/var.hpp>
#include <src/var/derived/unary_var_body.hpp>
#include <src/var/derived/multiply_var_body.hpp>

namespace nomad {

  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    operator*(const var<autodiff_order, strict_smoothness>& v1,
              const var<autodiff_order, strict_smoothness>& v2) {

    next_inputs_delta = 2;
    // next_partials_delta not used by multiply_var_body
    
    new multiply_var_body<autodiff_order>();
    
    push_dual_numbers<autodiff_order>(v1.first_val() * v2.first_val());
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    operator*(double v1,
              const var<autodiff_order, strict_smoothness>& v2) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    push_dual_numbers<autodiff_order>(v1 * v2.first_val());
    
    push_inputs(v2.dual_numbers());
    
    if (autodiff_order >= 1) push_partials(v1);
    
    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    operator*(const var<autodiff_order, strict_smoothness>& v1,
              double v2) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();
    
    push_dual_numbers<autodiff_order>(v1.first_val() * v2);
    
    push_inputs(v1.dual_numbers());
    
    if (autodiff_order >= 1) push_partials(v2);
    
    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }
  
}

#endif
