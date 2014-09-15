#ifndef nomad__src__scalar__functions__smooth_functions__acos_hpp
#define nomad__src__scalar__functions__smooth_functions__acos_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>

namespace nomad {

  inline double acos(double x) { return std::acos(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    acos(const var<AutodiffOrder, StrictSmoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_node<AutodiffOrder, partials_order>::n_partials();
    
    new unary_var_node<AutodiffOrder, partials_order>();

    const double x = input.first_val();
    push_dual_numbers<AutodiffOrder>(std::acos(x));
    
    push_inputs(input.dual_numbers());
    
    const double d1 = 1.0 / (1 - x * x);
    const double d2 = std::sqrt(d1);
    
    if (AutodiffOrder >= 1) push_partials(-d2);
    if (AutodiffOrder >= 2) push_partials(-x * d1 * d2);
    if (AutodiffOrder >= 3) push_partials(-(1 + 2 * x * x) * d1 * d1 * d2);

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
