#ifndef nomad__src__scalar__functions__smooth_functions__inv_hpp
#define nomad__src__scalar__functions__smooth_functions__inv_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double inv(double x) {
    return 1.0 / x;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    inv(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "inv");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double val = 1.0 / input.first_val();
    
    push_dual_numbers<AutodiffOrder>(val);
    
    push_inputs(input.dual_numbers());
    
    double dv = -val * val;
    
    if (AutodiffOrder >= 1) push_partials(dv);
    if (AutodiffOrder >= 2) push_partials(dv *= -2 * val);
    if (AutodiffOrder >= 3) push_partials(dv *= -3 * val);

    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_body_idx_ - 1);
    
  }

}

#endif
