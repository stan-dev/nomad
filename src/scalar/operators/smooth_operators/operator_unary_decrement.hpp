#ifndef nomad__src__scalar__operators__smooth_operators__operator_unary_decrement_hpp
#define nomad__src__scalar__operators__smooth_operators__operator_unary_decrement_hpp

#include <src/var/var.hpp>
#include <src/var/derived/unary_plus_var_node.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>&
    operator--(var<AutodiffOrder, StrictSmoothness>& v1) {
    
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    // next_partials_delta not used by unary_plus_var_node
    
    new unary_plus_var_node<AutodiffOrder>();
    
    push_dual_numbers<AutodiffOrder>(v1.first_val() - 1.0);
    
    push_inputs(v1.dual_numbers());

    v1.set_body(next_body_idx_ - 1);
    return v1;
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    operator--(const var<AutodiffOrder, StrictSmoothness>& v1, int /* dummy */) {

    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    // next_partials_delta not used by unary_plus_var_node
    
    new unary_plus_var_node<AutodiffOrder>();
    
    push_dual_numbers<AutodiffOrder>(v1.first_val() - 1.0);
    
    push_inputs(v1.dual_numbers());
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
