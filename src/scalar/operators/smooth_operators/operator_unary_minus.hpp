#ifndef nomad__src__scalar__operators__smooth_operators__operator_unary_minus_hpp
#define nomad__src__scalar__operators__smooth_operators__operator_unary_minus_hpp

#include <src/var/var.hpp>
#include <src/var/derived/unary_minus_var_node.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    operator-(const var<AutodiffOrder, StrictSmoothness>& v1) {

    create_node<unary_minus_var_node<AutodiffOrder>>(1);
    
    push_dual_numbers<AutodiffOrder>(-v1.first_val());
    
    push_inputs(v1.dual_numbers());
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}
#endif
