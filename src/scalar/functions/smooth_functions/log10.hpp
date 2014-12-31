#ifndef nomad__src__scalar__functions__smooth_functions__log10_hpp
#define nomad__src__scalar__functions__smooth_functions__log10_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double log10(double x) { return std::log10(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    log10(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) {
      double val = input.first_val();
      validate_input(val, "log10");
      validate_lower_bound(val, 0, "log10");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double val = input.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(log10(val));
    } catch (nomad_error) {
      throw nomad_output_value_error("log10");
    }
      
    push_inputs(input.dual_numbers());
    
    double val_inv = 1.0 / val;
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(val = val_inv * 4.48142011772455);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(val *= - val_inv);
      if (AutodiffOrder >= 3) push_partials<ValidateIO>(val *= - 2.0 * val_inv);
    } catch (nomad_error) {
      throw nomad_output_partial_error("log10");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
