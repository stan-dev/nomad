#ifndef nomad__src__scalar__functions__smooth_functions__binary_prod_cubes_hpp
#define nomad__src__scalar__functions__smooth_functions__binary_prod_cubes_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/binary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double binary_prod_cubes(double x, double y) {
    return x * x * x * y * y * y;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    binary_prod_cubes(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
                      const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {

    if (ValidateIO) {
      validate_input(v1.first_val(), "binary_prod_cubes");
      validate_input(v2.first_val(), "binary_prod_cubes");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 2;
    
    create_node<binary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double x = v1.first_val();
    double y = v2.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(binary_prod_cubes(x, y));
    } catch (nomad_error) {
      throw nomad_output_value_error("binary_prod_cubes");
    }
      
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    try {
      if (AutodiffOrder >= 1) {
        push_partials<ValidateIO>(3 * x * x * y * y * y);
        push_partials<ValidateIO>(3 * x * x * x * y * y);
      }
      if (AutodiffOrder >= 2) {
        push_partials<ValidateIO>(6 * x * y * y * y);
        push_partials<ValidateIO>(9 * x * x * y * y);
        push_partials<ValidateIO>(6 * x * x * x * y);
      }
      if (AutodiffOrder >= 3) {
        push_partials<ValidateIO>(6 * y * y * y);
        push_partials<ValidateIO>(18 * x * y * y);
        push_partials<ValidateIO>(18 * x * x * y);
        push_partials<ValidateIO>(6 * x * x * x);
      }
    } catch (nomad_error) {
      throw nomad_output_partial_error("binary_prod_cubes");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
