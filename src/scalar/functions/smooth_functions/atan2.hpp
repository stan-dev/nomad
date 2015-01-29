#ifndef nomad__src__scalar__functions__smooth_functions__atan2_hpp
#define nomad__src__scalar__functions__smooth_functions__atan2_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/var/derived/binary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double atan2(double y, double x) {
    return std::atan2(y, x);
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    atan2(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
          const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    
    if (ValidateIO) {
      validate_input(v1.first_val(), "atan2");
      validate_input(v2.first_val(), "atan2");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 2;
    
    create_node<binary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double y = v1.first_val();
    double x = v2.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(atan2(y, x));
    } catch (nomad_error) {
      throw nomad_output_value_error("atan2");
    }
      
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    double d = 1.0 / (x * x + y * y);
    
    try {
      if (AutodiffOrder >= 1) {
        push_partials<ValidateIO>(+ x * d);
        push_partials<ValidateIO>(- y * d);
      }
      if (AutodiffOrder >= 2) {
        double d2 = d * d;
        double p = 2 * x * y * d2;
        
        push_partials<ValidateIO>(-p);
        push_partials<ValidateIO>((y * y - x * x) * d2);
        push_partials<ValidateIO>(p);
      }
      if (AutodiffOrder >= 3) {
        double d3 = d * d * d;
        double p1 = 2 * y * (y * y - 3 * x * x) * d3;
        double p2 = 2 * x * (x * x - 3 * y * y) * d3;
        
        push_partials<ValidateIO>(- p2);
        push_partials<ValidateIO>(- p1);
        push_partials<ValidateIO>(p2);
        push_partials<ValidateIO>(p1);
      }
    } catch (nomad_error) {
      throw nomad_output_partial_error("atan2");
    }

    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    atan2(double y,
          const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    
    if (ValidateIO) {
      validate_input(y, "atan2");
      validate_input(v2.first_val(), "atan2");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v2.first_val();
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(atan2(y, x));
    } catch (nomad_error) {
      throw nomad_output_value_error("atan2");
    }
      
    push_inputs(v2.dual_numbers());
    
    double d = 1.0 / (x * x + y * y);
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(- y * d);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(2 * x * y * d * d);
      if (AutodiffOrder >= 3) push_partials<ValidateIO>(2 * y * (y * y - 3 * x * x) * d * d * d);
    } catch (nomad_error) {
      throw nomad_output_partial_error("atan2");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    atan2(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
          double x) {
    
    if (ValidateIO) {
      validate_input(v1.first_val(), "atan2");
      validate_input(x, "atan2");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double y = v1.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(atan2(y, x));
    } catch (nomad_error) {
      throw nomad_output_value_error("atan2");
    }
      
    push_inputs(v1.dual_numbers());
    
    double d = 1.0 / (x * x + y * y);
    
    try {
      if (AutodiffOrder >= 1) push_partials<ValidateIO>(x * d);
      if (AutodiffOrder >= 2) push_partials<ValidateIO>(- 2 * x * y * d * d);
      if (AutodiffOrder >= 3) push_partials<ValidateIO>(- 2 * x * (x * x - 3 * y * y) * d * d * d);
    } catch (nomad_error) {
      throw nomad_output_partial_error("atan2");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
