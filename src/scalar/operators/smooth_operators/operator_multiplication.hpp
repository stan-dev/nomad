#ifndef nomad__src__scalar__operators__smooth_operators__operator_multiplication_hpp
#define nomad__src__scalar__operators__smooth_operators__operator_multiplication_hpp

#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/var/derived/multiply_var_node.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    operator*(const var<AutodiffOrder, StrictSmoothness>& v1,
              const var<AutodiffOrder, StrictSmoothness>& v2) {

    next_inputs_delta = 2;
    // next_partials_delta not used by multiply_var_node
    
    new multiply_var_node<AutodiffOrder>();
    
    push_dual_numbers<AutodiffOrder>(v1.first_val() * v2.first_val());
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    operator*(double v1,
              const var<AutodiffOrder, StrictSmoothness>& v2) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_node<AutodiffOrder, partials_order>::n_partials();
    
    new unary_var_node<AutodiffOrder, partials_order>();

    push_dual_numbers<AutodiffOrder>(v1 * v2.first_val());
    
    push_inputs(v2.dual_numbers());
    
    if (AutodiffOrder >= 1) push_partials(v1);
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    operator*(const var<AutodiffOrder, StrictSmoothness>& v1,
              double v2) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_node<AutodiffOrder, partials_order>::n_partials();
    
    new unary_var_node<AutodiffOrder, partials_order>();
    
    push_dual_numbers<AutodiffOrder>(v1.first_val() * v2);
    
    push_inputs(v1.dual_numbers());
    
    if (AutodiffOrder >= 1) push_partials(v2);
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }
  
}

#endif
