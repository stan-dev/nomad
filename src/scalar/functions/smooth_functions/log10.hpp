#ifndef nomad__src__scalar__functions__smooth_functions__log10_hpp
#define nomad__src__scalar__functions__smooth_functions__log10_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>

namespace nomad {
  
  inline double log10(double x) { return std::log10(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    log10(const var<AutodiffOrder, StrictSmoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_node<AutodiffOrder, partials_order>::n_partials();
    
    new unary_var_node<AutodiffOrder, partials_order>();

    double val = input.first_val();
    
    push_dual_numbers<AutodiffOrder>(log10(val));
    
    push_inputs(input.dual_numbers());
    
    double val_inv = 1.0 / val;
    
    if (AutodiffOrder >= 1) push_partials(val = val_inv * 4.48142011772455);
    if (AutodiffOrder >= 2) push_partials(val *= - val_inv);
    if (AutodiffOrder >= 3) push_partials(val *= - 2.0 * val_inv);

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
