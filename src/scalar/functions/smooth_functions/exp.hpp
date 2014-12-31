#ifndef nomad__src__scalar__functions__smooth_functions__exp_hpp
#define nomad__src__scalar__functions__smooth_functions__exp_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double exp(double x) { return std::exp(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    exp(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "exp");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double val = std::exp(input.first_val());
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(val);
    } catch (nomad_error) {
      throw nomad_output_value_error("exp");
    }
      
    push_inputs(input.dual_numbers());
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(val);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(val);
      if (AutodiffOrder >= 3) push_partials<ValidateIO>(val);
    } catch (nomad_error) {
      throw nomad_output_partial_error("exp");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
