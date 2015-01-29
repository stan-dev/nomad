#ifndef nomad__src__scalar__functions__smooth_functions__tanh_hpp
#define nomad__src__scalar__functions__smooth_functions__tanh_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double tanh(double x) { return std::tanh(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    tanh(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "tanh");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double t = std::tanh(input.first_val());
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(t);
    } catch (nomad_error) {
      throw nomad_output_value_error("tanh");
    }
      
    push_inputs(input.dual_numbers());
    
    double sech2 = 1.0 / std::cosh(input.first_val());
    sech2 *= sech2;
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(sech2);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(-2 * sech2 * t);
      if (AutodiffOrder >= 3) push_partials<ValidateIO>(-2 * sech2 * sech2 + 4 * sech2 * t * t);
    } catch (nomad_error) {
      throw nomad_output_partial_error("tanh");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
