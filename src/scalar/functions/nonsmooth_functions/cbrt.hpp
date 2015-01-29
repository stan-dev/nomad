#ifndef nomad__src__scalar__functions__nonsmooth_functions__cbrt_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__cbrt_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double cbrt(double x) {
    return std::cbrt(x);
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    cbrt(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {

    if (ValidateIO) validate_input(input.first_val(), "cbrt");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double val = std::cbrt(input.first_val());
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(val);
    } catch (nomad_error) {
      throw nomad_output_value_error("cbrt");
    }
      
    push_inputs(input.dual_numbers());
    
    double d2 = 1.0 / input.first_val();

    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(val *= 1.0 * d2 / 3.0);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(val *= - 2.0 * d2 / 3.0);
      if (AutodiffOrder >= 3) push_partials<ValidateIO>(val *= - 5.0 * d2 / 3.0);
    } catch (nomad_error) {
      throw nomad_output_partial_error("cbrt");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
