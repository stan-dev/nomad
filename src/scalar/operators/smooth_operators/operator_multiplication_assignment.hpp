#ifndef nomad__src__scalar__operators__smooth_operators__operator_multiplication_assignment_hpp
#define nomad__src__scalar__operators__smooth_operators__operator_multiplication_assignment_hpp

#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/var/derived/multiply_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>&
    operator*=(var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
               const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {

    if (ValidateIO) {
      validate_input(v1.first_val(), "operator*=");
      validate_input(v2.first_val(), "operator*=");
    }
      
    create_node<multiply_var_node<AutodiffOrder>>(2);
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(v1.first_val() * v2.first_val());
    } catch (nomad_error) {
      throw nomad_output_value_error("operator*=");
    }
      
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    v1.set_node(next_node_idx_ - 1);
    return v1;
    
  }

  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>&
    operator*=(var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
               double v2) {
    
    if (ValidateIO) {
      validate_input(v1.first_val(), "operator*=");
      validate_input(v2, "operator*=");
    }
      
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(v1.first_val() * v2);
    } catch (nomad_error) {
      throw nomad_output_value_error("operator*=");
    }
      
    push_inputs(v1.dual_numbers());
    
    if (AutodiffOrder >= 1) push_partials<ValidateIO>(v2);
    
    v1.set_node(next_node_idx_ - 1);
    return v1;
    
  }
  
}

#endif
