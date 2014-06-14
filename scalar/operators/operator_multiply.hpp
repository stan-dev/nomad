#ifndef nomad__scalar__operators__operator_multiply_hpp
#define nomad__scalar__operators__operator_multiply_hpp

#include <var/var.hpp>
#include <var/derived/multiply_var_body.hpp>

namespace nomad {

  template <short autodiff_order>
  inline var<autodiff_order> operator*(const var<autodiff_order>& v1,
                                       const var<autodiff_order>& v2) {

    next_inputs_delta = 2;
    // next_partials_delta not used by multiply_var_body
    
    new multiply_var_body<autodiff_order>();
    
    push_dual_numbers<autodiff_order>(v1.first_val() * v2.first_val());
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

  template <short autodiff_order>
  inline var<autodiff_order> operator*(double v1,
                                       const var<autodiff_order>& v2) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      var_body<autodiff_order, partials_order>::n_partials(n_inputs);
    
    new var_body<autodiff_order, partials_order>(n_inputs);

    push_dual_numbers<autodiff_order>(v1 * v2.first_val());
    
    push_inputs(v2.dual_numbers());
    
    if (autodiff_order >= 1) push_partials(v1);
    
    return var<autodiff_order>(next_body_idx_ - 1);
    
  }
  
  template <short autodiff_order>
  inline var<autodiff_order> operator*(const var<autodiff_order>& v1,
                                       double v2) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      var_body<autodiff_order, partials_order>::n_partials(n_inputs);
    
    new var_body<autodiff_order, partials_order>(n_inputs);
    
    push_dual_numbers<autodiff_order>(v1.first_val() * v2);
    
    push_inputs(v1.dual_numbers());
    
    if (autodiff_order >= 1) push_partials(v2);
    
    return var<autodiff_order>(next_body_idx_ - 1);
    
  }
  
}

#endif
