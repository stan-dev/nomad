#ifndef nomad__src__scalar__functions__smooth_functions__inv_square_hpp
#define nomad__src__scalar__functions__smooth_functions__inv_square_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double inv_square(double x) {
    return 1.0 / (x * x);
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    inv_square(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "inv_square");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double d = 1.0 / input.first_val();
    double val = d * d;
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(val);
    } catch (nomad_error) {
      throw nomad_output_value_error("inv_square");
    }
      
    push_inputs(input.dual_numbers());
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(val *= -2 * d);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(val *= -3 * d);
      if (AutodiffOrder >= 3) push_partials<ValidateIO>(val *= -4 * d);
    } catch (nomad_error) {
      throw nomad_output_partial_error("inv_square");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
