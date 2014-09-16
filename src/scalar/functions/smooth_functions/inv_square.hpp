#ifndef nomad__src__scalar__functions__smooth_functions__inv_square_hpp
#define nomad__src__scalar__functions__smooth_functions__inv_square_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double inv_square(double x) {
    return 1.0 / (x * x);
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    inv_square(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "inv_square");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double d = 1.0 / input.first_val();
    double val = d * d;
    
    push_dual_numbers<AutodiffOrder>(val);
    
    push_inputs(input.dual_numbers());
    
    if (AutodiffOrder >= 1) push_partials(val *= -2 * d);
    if (AutodiffOrder >= 2) push_partials(val *= -3 * d);
    if (AutodiffOrder >= 3) push_partials(val *= -4 * d);

    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
