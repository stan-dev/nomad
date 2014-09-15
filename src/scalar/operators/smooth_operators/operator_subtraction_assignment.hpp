#ifndef nomad__src__scalar__operators__smooth_operators__operator_subtraction_assignment_hpp
#define nomad__src__scalar__operators__smooth_operators__operator_subtraction_assignment_hpp

#include <src/var/var.hpp>
#include <src/var/derived/unary_minus_var_node.hpp>
#include <src/var/derived/unary_plus_var_node.hpp>
#include <src/var/derived/binary_minus_var_node.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>&
    operator-=(var<AutodiffOrder, StrictSmoothness>& v1,
               const var<AutodiffOrder, StrictSmoothness>& v2) {

    create_node<binary_minus_var_node<AutodiffOrder>>(2);
    
    push_dual_numbers<AutodiffOrder>(v1.first_val() - v2.first_val());
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    v1.set_body(next_body_idx_ - 1);
    return v1;
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>&
    operator-=(var<AutodiffOrder, StrictSmoothness>& v1,
               double y) {
    
    create_node<unary_plus_var_node<AutodiffOrder>>(1);
    
    push_dual_numbers<AutodiffOrder>(v1.first_val() - y);
    
    push_inputs(v1.dual_numbers());
    
    v1.set_body(next_body_idx_ - 1);
    return v1;
    
  }

}
#endif
