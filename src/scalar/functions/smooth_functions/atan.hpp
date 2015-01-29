#ifndef nomad__src__scalar__functions__smooth_functions__atan_hpp
#define nomad__src__scalar__functions__smooth_functions__atan_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {

  inline double atan(double x) { return std::atan(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    atan(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "atan");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    const double x = input.first_val();
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(std::atan(x));
    } catch (nomad_error) {
      throw nomad_output_value_error("atan");
    }
      
    push_inputs(input.dual_numbers());
    
    const double d1 = 1.0 / (1 + x * x);
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(d1);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(- 2 * x * d1 * d1);
      if (AutodiffOrder >= 3) push_partials<ValidateIO>((- 2 + 6 * x * x) * d1 * d1 * d1);
    } catch (nomad_error) {
      throw nomad_output_partial_error("atan");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
