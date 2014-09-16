#ifndef nomad__src__scalar__functions__smooth_functions__inv_sqrt_hpp
#define nomad__src__scalar__functions__smooth_functions__inv_sqrt_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double inv_sqrt(double x) {
    return 1.0 / std::sqrt(x);
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    inv_sqrt(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) {
      double val = input.first_val();
      validate_input(val, "inv_sqrt");
      validate_lower_bound(val, 0, "inv_sqrt");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double d = 1.0 / input.first_val();
    double sqrtd = sqrt(d);
    
    push_dual_numbers<AutodiffOrder>(sqrtd);
    
    push_inputs(input.dual_numbers());
    
    if (AutodiffOrder >= 1) push_partials(-0.5 * d * sqrtd);
    if (AutodiffOrder >= 2) push_partials(0.75 * d * d * sqrtd);
    if (AutodiffOrder >= 3) push_partials(-1.875 * d * d * d * sqrtd);

    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
