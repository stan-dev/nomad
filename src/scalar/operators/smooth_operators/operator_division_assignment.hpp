#ifndef nomad__src__scalar__operators__smooth_operators__operator_division_assignment_hpp
#define nomad__src__scalar__operators__smooth_operators__operator_division_assignment_hpp

#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/var/derived/binary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>&
    operator/=(var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
               const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {

    if (ValidateIO) {
      validate_input(v1.first_val(), "operator/=");
      validate_input(v2.first_val(), "operator/=");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 2;
    
    create_node<binary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v1.first_val();
    double y_inv = 1.0 / v2.first_val();
    double val = x * y_inv;
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(val);
    } catch (nomad_error) {
      throw nomad_output_value_error("operator/=");
    }
      
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    double y_inv_n = y_inv * y_inv;
    
    if (AutodiffOrder >= 1) {
      push_partials<ValidateIO>(y_inv);
      push_partials<ValidateIO>(- val * y_inv);
    }
    if (AutodiffOrder >= 2) {
      push_partials<ValidateIO>(0);
      push_partials<ValidateIO>(-y_inv_n);
      push_partials<ValidateIO>(2 * val * y_inv_n);
    }
    if (AutodiffOrder >= 3) {
      y_inv_n *= y_inv;
      push_partials<ValidateIO>(0);
      push_partials<ValidateIO>(0);
      push_partials<ValidateIO>(2 * y_inv_n);
      push_partials<ValidateIO>(-6 * val * y_inv_n);
    }
    
    v1.set_node(next_node_idx_ - 1);
    return v1;
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>&
    operator/=(var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
               double y) {
    
    if (ValidateIO) {
      validate_input(v1.first_val(), "operator/=");
      validate_input(y, "operator/=");
    }
      
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v1.first_val();
    double y_inv = 1.0 / y;
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(x * y_inv);
    } catch (nomad_error) {
      throw nomad_output_value_error("operator/=");
    }
      
    push_inputs(v1.dual_numbers());
    
    if (AutodiffOrder >= 1) push_partials<ValidateIO>(y_inv);
    
    v1.set_node(next_node_idx_ - 1);
    return v1;
    
  }

}
#endif
