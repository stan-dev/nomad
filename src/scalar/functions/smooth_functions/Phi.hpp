#ifndef nomad__src__scalar__functions__smooth_functions__Phi_hpp
#define nomad__src__scalar__functions__smooth_functions__Phi_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double Phi(double x) {
    if (x < -40)
      return 0;
    else if (x < 0)
      return 0.5 * std::erfc(-0.70710678118655 * x);
    else if (x < 40)
      return 0.5 * (1.0 + std::erf(0.70710678118655 * x));
    else
      return 1;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    Phi(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "Phi");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double x = input.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(Phi(x));
    } catch (nomad_error) {
      throw nomad_output_value_error("Phi");
    }
      
    push_inputs(input.dual_numbers());
    
    double C = 0.39894228040143 * exp(- 0.5 * x * x);
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(C);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(- x * C);
      if (AutodiffOrder >= 3) push_partials<ValidateIO>((x * x - 1) * C);
    } catch (nomad_error) {
      throw nomad_output_partial_error("Phi");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
