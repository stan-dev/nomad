#ifndef nomad__src__scalar__operators__smooth_operators__operator_unary_minus_hpp
#define nomad__src__scalar__operators__smooth_operators__operator_unary_minus_hpp

#include <src/var/var.hpp>
#include <src/var/derived/unary_minus_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    operator-(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1) {

    if (ValidateIO) validate_input(v1.first_val(), "operator-");
      
    create_node<unary_minus_var_node<AutodiffOrder>>(1);
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(-v1.first_val());
    } catch (nomad_error) {
      throw nomad_output_value_error("operator-");
    }
      
    push_inputs(v1.dual_numbers());
    
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}
#endif
