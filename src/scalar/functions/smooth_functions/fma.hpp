#ifndef nomad__src__scalar__functions__smooth_functions__fma_hpp
#define nomad__src__scalar__functions__smooth_functions__fma_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double fma(double x, double y, double z) {
    return std::fma(x, y, z);
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    fma(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
        const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2,
        const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v3) {
    
    if (ValidateIO) {
      validate_input(v1.first_val(), "fma");
      validate_input(v2.first_val(), "fma");
      validate_input(v3.first_val(), "fma");
    }
      
    const short partials_order = 2;
    const unsigned int n_inputs = 3;
 
    create_node<var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v1.first_val();
    double y = v2.first_val();
    double z = v3.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(fma(x, y, z));
    } catch (nomad_error) {
      throw nomad_output_value_error("fma");
    }
      
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    push_inputs(v3.dual_numbers());
    
    try {
      if (AutodiffOrder >= 1) {
        push_partials<ValidateIO>(y);
        push_partials<ValidateIO>(x);
        push_partials<ValidateIO>(1);
      }
      if (AutodiffOrder >= 2) {
        push_partials<ValidateIO>(0);
        push_partials<ValidateIO>(1);
        push_partials<ValidateIO>(0);
        
        push_partials<ValidateIO>(0);
        push_partials<ValidateIO>(0);
        push_partials<ValidateIO>(0);
      }
    } catch (nomad_error) {
      throw nomad_output_partial_error("fma");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
