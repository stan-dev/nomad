#ifndef nomad__src__scalar__functions__smooth_functions__inv_logit_hpp
#define nomad__src__scalar__functions__smooth_functions__inv_logit_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double inv_logit(double x) {
    if (x > 0)
      return 1.0 / (1.0 + exp(-x));
    else {
      double e = std::exp(x);
      return e / (1.0 + e);
    }
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    inv_logit(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "inv_logit");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double s = inv_logit(input.first_val());
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(s);
    } catch (nomad_error) {
      throw nomad_output_value_error("inv_logit");
    }
      
    push_inputs(input.dual_numbers());
    
    double ds = s * (1 - s);
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(ds);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(ds * (1 - 2 * s) );
      if (AutodiffOrder >= 3) push_partials<ValidateIO>(ds * (1 - 6 * s * (1 - s)) );
    } catch (nomad_error) {
      throw nomad_output_partial_error("inv_logit");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
