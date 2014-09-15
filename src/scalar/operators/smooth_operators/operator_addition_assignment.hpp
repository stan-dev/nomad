#ifndef nomad__src__scalar__operators__smooth_operators__operator_addition_assignment_hpp
#define nomad__src__scalar__operators__smooth_operators__operator_addition_assignment_hpp

#include <src/var/var.hpp>
#include <src/var/derived/unary_plus_var_body.hpp>
#include <src/var/derived/binary_sum_var_body.hpp>

namespace nomad {

  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>&
    operator+=(var<autodiff_order, strict_smoothness>& v1,
               const var<autodiff_order, strict_smoothness>& v2) {

    create_node<binary_sum_var_body<autodiff_order>>(2);
      
    push_dual_numbers<autodiff_order>(v1.first_val() + v2.first_val());
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    v1.set_body(next_body_idx_ - 1);
    return v1;
    
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>&
    operator+=(var<autodiff_order, strict_smoothness>& v1,
               double y) {
    
    create_node<unary_plus_var_body<autodiff_order>>(1);
      
    push_dual_numbers<autodiff_order>(v1.first_val() + y);
    
    push_inputs(v1.dual_numbers());

    v1.set_body(next_body_idx_ - 1);
    return v1;
    
  }

}
#endif
