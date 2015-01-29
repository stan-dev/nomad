#ifndef nomad__src__scalar__operators__smooth_operators__operator_addition_hpp
#define nomad__src__scalar__operators__smooth_operators__operator_addition_hpp

#include <src/var/var.hpp>
#include <src/var/derived/unary_plus_var_node.hpp>
#include <src/var/derived/binary_sum_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    operator+(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
              const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {

    if (ValidateIO) {
      validate_input(v1.first_val(), "operator+");
      validate_input(v2.first_val(), "operator+");
    }
      
    create_node<binary_sum_var_node<AutodiffOrder>>(2);
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(v1.first_val() + v2.first_val());
    } catch (nomad_error) {
      throw nomad_output_value_error("operator+");
    }
      
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    operator+(double x,
              const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    
    if (ValidateIO) {
      validate_input(x, "operator+");
      validate_input(v2.first_val(), "operator+");
    }
      
    create_node<unary_plus_var_node<AutodiffOrder>>(1);
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(x + v2.first_val());
    } catch (nomad_error) {
      throw nomad_output_value_error("operator+");
    }
      
    push_inputs(v2.dual_numbers());
    
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    operator+(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
              double y) {
    
    if (ValidateIO) {
      validate_input(v1.first_val(), "operator+");
      validate_input(y, "operator+");
    }
      
    create_node<unary_plus_var_node<AutodiffOrder>>(1);
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(v1.first_val() + y);
    } catch (nomad_error) {
      throw nomad_output_value_error("operator+");
    }
      
    push_inputs(v1.dual_numbers());
    
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}
#endif
