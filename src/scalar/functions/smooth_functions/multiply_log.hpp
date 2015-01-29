#ifndef nomad__src__scalar__functions__smooth_functions__multiply_log_hpp
#define nomad__src__scalar__functions__smooth_functions__multiply_log_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/var/derived/binary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double multiply_log(double x, double y) {
    if (x == 0.0 && y == 0.0) return 0.0;
    return x * std::log(y);
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    multiply_log(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
                 const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    
    if (ValidateIO) {
      validate_input(v1.first_val(), "multiply_log");
      
      double val1 = v2.first_val();
      validate_input(val1, "multiply_log");
      validate_lower_bound(val1, 0, "multiply_log");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 2;
    
    create_node<binary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v1.first_val();
    double y = v2.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(multiply_log(x, y));
    } catch (nomad_error) {
      throw nomad_output_value_error("multiply_log");
    }
      
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    double y_inv = 1.0 / y;
    
    try {
      if (AutodiffOrder >= 1) {
        push_partials<ValidateIO>(std::log(y));
        push_partials<ValidateIO>(x * y_inv);
      }
      if (AutodiffOrder >= 2) {
        push_partials<ValidateIO>(0);
        push_partials<ValidateIO>(y_inv);
        push_partials<ValidateIO>(- x * y_inv * y_inv);
      }
      if (AutodiffOrder >= 3) {
        push_partials<ValidateIO>(0);
        push_partials<ValidateIO>(0);
        push_partials<ValidateIO>(- y_inv * y_inv);
        push_partials<ValidateIO>(2 * x * y_inv * y_inv * y_inv);
      }
    } catch (nomad_error) {
      throw nomad_output_partial_error("multiply_log");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    multiply_log(double x,
                 const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    
    if (ValidateIO) {
      validate_input(x, "multiply_log");
      
      double val1 = v2.first_val();
      validate_input(val1, "multiply_log");
      validate_lower_bound(val1, 0, "multiply_log");
    }
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double y = v2.first_val();
    
    push_dual_numbers<AutodiffOrder, ValidateIO>(multiply_log(x, y));
    
    push_inputs(v2.dual_numbers());
    
    double y_inv = 1.0 / y;
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(x * y_inv);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(- x * y_inv * y_inv);
      if (AutodiffOrder >= 3) push_partials<ValidateIO>(2 * x * y_inv * y_inv * y_inv);
    } catch (nomad_error) {
      throw nomad_output_partial_error("multiply_log");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    multiply_log(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
                 double y) {
    
    if (ValidateIO) {
      validate_input(v1.first_val(), "multiply_log");
      
      validate_input(y, "multiply_log");
      validate_lower_bound(y, 0, "multiply_log");
    }
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v1.first_val();
    
    push_dual_numbers<AutodiffOrder, ValidateIO>(multiply_log(x, y));
    
    push_inputs(v1.dual_numbers());
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(std::log(y));
    } catch (nomad_error) {
      throw nomad_output_partial_error("multiply_log");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
