#ifndef nomad__src__scalar__functions__smooth_functions__pow_hpp
#define nomad__src__scalar__functions__smooth_functions__pow_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/var/derived/binary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double pow(double x, double y) { return std::pow(x, y); }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    pow(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
        const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    
    if (ValidateIO) {
      double val1 = v1.first_val();
      validate_input(val1, "pow");
      
      double val2 = v2.first_val();
      validate_input(val2, "pow");
      
      if (std::floor(val2) != val2)
        validate_lower_bound(val1, 0, "pow");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 2;
    
    create_node<binary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v1.first_val();
    double y = v2.first_val();
    double val = std::pow(x, y);
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(val);
    } catch (nomad_error) {
      throw nomad_output_value_error("pow");
    }
      
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    double lx = std::log(x);
      
    try {
      if (AutodiffOrder >= 1) {
        push_partials<ValidateIO>(val * y / x);
        push_partials<ValidateIO>(val * lx);
      }
      if (AutodiffOrder >= 2) {
        push_partials<ValidateIO>( y * (y - 1.0) * val / (x * x) );
        push_partials<ValidateIO>( (1 + y * lx) * val / x );
        push_partials<ValidateIO>( lx * lx * val );
      }
      if (AutodiffOrder >= 3) {
        push_partials<ValidateIO>((y - 2.0) * (y - 1.0) * y * val / (x  * x * x) );
        push_partials<ValidateIO>( ((y - 1.0) * y * lx + 2 * y - 1.0) * val / (x * x) );
        push_partials<ValidateIO>( lx * (2.0 + y * lx) * val / x );
        push_partials<ValidateIO>( lx * lx * lx * val );
      }
    } catch (nomad_error) {
      throw nomad_output_partial_error("pow");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    pow(double x,
        const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    
    if (ValidateIO) {
      validate_input(x, "pow");
      
      double val2 = v2.first_val();
      validate_input(val2, "pow");
      
      if (std::floor(val2) != val2)
        validate_lower_bound(x, 0, "pow");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double y = v2.first_val();
    double val = std::pow(x, y);
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(val);
    } catch (nomad_error) {
      throw nomad_output_value_error("pow");
    }
      
    push_inputs(v2.dual_numbers());
    
    double lx = std::log(x);
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(val * lx);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(lx * lx * val);
      if (AutodiffOrder >= 3) push_partials<ValidateIO>(lx * lx * lx * val);
    } catch (nomad_error) {
      throw nomad_output_partial_error("pow");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    pow(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
        double y) {
    
    if (ValidateIO) {
      double val1 = v1.first_val();
      validate_input(val1, "pow");
      
      validate_input(y, "pow");
      
      if (std::floor(y) != y)
        validate_lower_bound(val1, 0, "pow");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v1.first_val();
    double val = std::pow(x, y);
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(val);
    } catch (nomad_error) {
      throw nomad_output_value_error("pow");
    }
      
    push_inputs(v1.dual_numbers());
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(val * y / x);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>( y * (y - 1.0) * val / (x * x) );
      if (AutodiffOrder >= 3) push_partials<ValidateIO>((y - 2.0) * (y - 1.0) * y * val / (x  * x * x) );
    } catch (nomad_error) {
      throw nomad_output_partial_error("pow");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
