#ifndef nomad__src__scalar__functions__smooth_functions__inv_sqrt_hpp
#define nomad__src__scalar__functions__smooth_functions__inv_sqrt_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double inv_sqrt(double x) {
    return 1.0 / std::sqrt(x);
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    inv_sqrt(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) {
      double val = input.first_val();
      validate_input(val, "inv_sqrt");
      validate_lower_bound(val, 0, "inv_sqrt");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double d = 1.0 / input.first_val();
    double sqrtd = sqrt(d);
      
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(sqrtd);
    } catch (nomad_error) {
      throw nomad_output_value_error("inv_sqrt");
    }
      
    push_inputs(input.dual_numbers());
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(-0.5 * d * sqrtd);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(0.75 * d * d * sqrtd);
      if (AutodiffOrder >= 3) push_partials<ValidateIO>(-1.875 * d * d * d * sqrtd);
    } catch (nomad_error) {
      throw nomad_output_partial_error("inv_sqrt");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
