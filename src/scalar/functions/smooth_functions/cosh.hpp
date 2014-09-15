#ifndef nomad__src__scalar__functions__smooth_functions__cosh_hpp
#define nomad__src__scalar__functions__smooth_functions__cosh_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>

namespace nomad {
  
  inline double cosh(double x) { return std::cosh(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    cosh(const var<AutodiffOrder, StrictSmoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_node<AutodiffOrder, partials_order>::n_partials();
    
    new unary_var_node<AutodiffOrder, partials_order>();

    double c = std::cosh(input.first_val());
    double s = std::sinh(input.first_val());
    
    push_dual_numbers<AutodiffOrder>(c);
    
    push_inputs(input.dual_numbers());
    
    if (AutodiffOrder >= 1) push_partials(s);
    if (AutodiffOrder >= 2) push_partials(c);
    if (AutodiffOrder >= 3) push_partials(s);

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
