#ifndef nomad__src__scalar__functions__smooth_functions__square_hpp
#define nomad__src__scalar__functions__smooth_functions__square_hpp

#include <src/var/var.hpp>
#include <src/var/derived/square_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {

  inline double square(double input) {
    return input * input;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    square(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "square");
      
    create_node<square_var_node<AutodiffOrder>>(1);
    
    double val = input.first_val();

    push_dual_numbers<AutodiffOrder>(val * val);
    push_inputs(input.dual_numbers());
    
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_body_idx_ - 1);
    
  }

}

#endif
