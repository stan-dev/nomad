#ifndef nomad__scalar__operators__operator_addition_hpp
#define nomad__scalar__operators__operator_addition_hpp

#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>
#include <var/derived/binary_sum_var_body.hpp>

namespace nomad {

  template <short autodiff_order>
  inline var<autodiff_order> operator+(const var<autodiff_order>& v1,
                                       const var<autodiff_order>& v2) {

    const unsigned int n_inputs = 2;
    
    next_inputs_delta = n_inputs;
    // next_partials_delta not used by binary_sum_var_body
    
    new binary_sum_var_body<autodiff_order>();
    
    push_dual_numbers<autodiff_order>(v1.first_val() + v2.first_val());
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    return var<autodiff_order>(next_body_idx_ - 1);
    
  }
  
  template <short autodiff_order>
  inline var<autodiff_order> operator+(double x,
                                       const var<autodiff_order>& v2) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();
    
    push_dual_numbers<autodiff_order>(x + v2.first_val());
    
    push_inputs(v2.dual_numbers());
    
    if (autodiff_order >= 1) push_partials(1);
    if (autodiff_order >= 2) push_partials(0);
    if (autodiff_order >= 3) push_partials(0);
    
    return var<autodiff_order>(next_body_idx_ - 1);
    
  }
  
  template <short autodiff_order>
  inline var<autodiff_order> operator+(const var<autodiff_order>& v1,
                                       double y) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();
    
    push_dual_numbers<autodiff_order>(v1.first_val() + y);
    
    push_inputs(v1.dual_numbers());
    
    if (autodiff_order >= 1) push_partials(1);
    if (autodiff_order >= 2) push_partials(0);
    if (autodiff_order >= 3) push_partials(0);
    
    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}
#endif
