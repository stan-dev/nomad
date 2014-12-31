#ifndef nomad__src__scalar__functions__smooth_functions__log1p_hpp
#define nomad__src__scalar__functions__smooth_functions__log1p_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double log1p(double x) { return std::log1p(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    log1p(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) {
      double val = input.first_val();
      validate_input(val, "log1p");
      validate_lower_bound(val, -1, "log1p");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double val = input.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(log1p(val));
    } catch (nomad_error) {
      throw nomad_output_value_error("log1p");
    }
      
    push_inputs(input.dual_numbers());
    
    double val_inv = 1.0 / (1 + val);
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(val = val_inv);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(val *= - val_inv);
      if (AutodiffOrder >= 3) push_partials<ValidateIO>(val *= - 2.0 * val_inv);
    } catch (nomad_error) {
      throw nomad_output_partial_error("log1p");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
