#ifndef nomad__src__scalar__functions__smooth_functions__cbrt_hpp
#define nomad__src__scalar__functions__smooth_functions__cbrt_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>

namespace nomad {
  
  inline double cbrt(double x) {
    return std::cbrt(x);
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    cbrt(const var<AutodiffOrder, StrictSmoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_node<AutodiffOrder, partials_order>::n_partials();
    
    new unary_var_node<AutodiffOrder, partials_order>();

    double val = std::cbrt(input.first_val());
    
    push_dual_numbers<AutodiffOrder>(val);
    
    push_inputs(input.dual_numbers());
    
    double d2 = 1.0 / input.first_val();
    
    if (AutodiffOrder >= 1) push_partials(val *= 1.0 * d2 / 3.0);
    if (AutodiffOrder >= 2) push_partials(val *= - 2.0 * d2 / 3.0);
    if (AutodiffOrder >= 3) push_partials(val *= - 5.0 * d2 / 3.0);

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
