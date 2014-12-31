#ifndef nomad__src__scalar__functions__smooth_functions__lgamma_hpp
#define nomad__src__scalar__functions__smooth_functions__lgamma_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/scalar/functions/smooth_functions/polygamma.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double lgamma(double x) { return std::lgamma(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    lgamma(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "lgamma");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double val = input.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(lgamma(val));
    } catch (nomad_error) {
      throw nomad_output_value_error("lgamma");
    }
      
    push_inputs(input.dual_numbers());
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(digamma(val));
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(trigamma(val));
      if (AutodiffOrder >= 3) push_partials<ValidateIO>(quadrigamma(val));
    } catch (nomad_error) {
      throw nomad_output_partial_error("lgamma");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
