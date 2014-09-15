#ifndef nomad__src__scalar__operators__smooth_operators__operator_subtraction_hpp
#define nomad__src__scalar__operators__smooth_operators__operator_subtraction_hpp

#include <src/var/var.hpp>
#include <src/var/derived/unary_minus_var_node.hpp>
#include <src/var/derived/unary_plus_var_node.hpp>
#include <src/var/derived/binary_minus_var_node.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    operator-(const var<AutodiffOrder, StrictSmoothness>& v1,
              const var<AutodiffOrder, StrictSmoothness>& v2) {

    create_node<binary_minus_var_node<AutodiffOrder>>(2);
    
    push_dual_numbers<AutodiffOrder>(v1.first_val() - v2.first_val());
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    operator-(double x,
              const var<AutodiffOrder, StrictSmoothness>& v2) {
    
    create_node<unary_minus_var_node<AutodiffOrder>>(1);
    
    push_dual_numbers<AutodiffOrder>(x - v2.first_val());
    
    push_inputs(v2.dual_numbers());
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    operator-(const var<AutodiffOrder, StrictSmoothness>& v1,
              double y) {
    
    create_node<unary_plus_var_node<AutodiffOrder>>(1);
    
    push_dual_numbers<AutodiffOrder>(v1.first_val() - y);
    
    push_inputs(v1.dual_numbers());

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}
#endif
