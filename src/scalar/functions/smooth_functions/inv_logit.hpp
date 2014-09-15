#ifndef nomad__src__scalar__functions__smooth_functions__inv_logit_hpp
#define nomad__src__scalar__functions__smooth_functions__inv_logit_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>

namespace nomad {
  
  inline double inv_logit(double x) {
    if (x > 0)
      return 1.0 / (1.0 + exp(-x));
    else {
      double e = std::exp(x);
      return e / (1.0 + e);
    }
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    inv_logit(const var<AutodiffOrder, StrictSmoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_node<AutodiffOrder, partials_order>::n_partials();
    
    new unary_var_node<AutodiffOrder, partials_order>();

    double s = inv_logit(input.first_val());
    
    push_dual_numbers<AutodiffOrder>(s);
    
    push_inputs(input.dual_numbers());
    
    double ds = s * (1 - s);
    
    if (AutodiffOrder >= 1) push_partials(ds);
    if (AutodiffOrder >= 2) push_partials(ds * (1 - 2 * s) );
    if (AutodiffOrder >= 3) push_partials(ds * (1 - 6 * s * (1 - s)) );

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
