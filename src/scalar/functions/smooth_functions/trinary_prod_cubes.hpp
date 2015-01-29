#ifndef nomad__src__scalar__functions__smooth_functions__trinary_prod_cubes_hpp
#define nomad__src__scalar__functions__smooth_functions__trinary_prod_cubes_hpp

#include <src/var/var.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double trinary_prod_cubes(double x, double y, double z) {
    return x * x * x * y * y * y * z * z * z;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    trinary_prod_cubes(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
                       const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2,
                       const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v3) {

    if (ValidateIO) {
      validate_input(v1.first_val(), "trinary_prod_cubes");
      validate_input(v2.first_val(), "trinary_prod_cubes");
      validate_input(v3.first_val(), "trinary_prod_cubes");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 3;

    create_node<var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v1.first_val();
    double y = v2.first_val();
    double z = v3.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(trinary_prod_cubes(x, y, z));
    } catch (nomad_error) {
      throw nomad_output_value_error("trinary_prod_cubes");
    }
      
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    push_inputs(v3.dual_numbers());
    
    try {
      if (AutodiffOrder >= 1) {
        push_partials<ValidateIO>(3 * x * x * y * y * y * z * z * z);
        push_partials<ValidateIO>(3 * x * x * x * y * y * z * z * z);
        push_partials<ValidateIO>(3 * x * x * x * y * y * y * z * z);
      }
      if (AutodiffOrder >= 2) {
        push_partials<ValidateIO>(6 * x * y * y * y * z * z * z);
        push_partials<ValidateIO>(9 * x * x * y * y * z * z * z);
        push_partials<ValidateIO>(6 * x * x * x * y * z * z * z);
        
        push_partials<ValidateIO>(9 * x * x * y * y * y * z * z);
        push_partials<ValidateIO>(9 * x * x * x * y * y * z * z);
        push_partials<ValidateIO>(6 * x * x * x * y * y * y * z);
      }
      if (AutodiffOrder >= 3) {
        push_partials<ValidateIO>(6 * y * y * y * z * z * z);
        push_partials<ValidateIO>(18 * x * y * y * z * z * z);
        push_partials<ValidateIO>(18 * x * x * y * z * z * z);
        push_partials<ValidateIO>(6 * x * x * x * z * z * z);
        push_partials<ValidateIO>(18 * x * y * y * y * z * z);
        push_partials<ValidateIO>(27 * x * x * y * y * z * z);
        push_partials<ValidateIO>(18 * x * x * x * y * z * z);
        push_partials<ValidateIO>(18 * x * x * y * y * y * z);
        push_partials<ValidateIO>(18 * x * x * x * y * y * z);
        push_partials<ValidateIO>(6 * x * x * x * y * y * y);
      }
    } catch (nomad_error) {
      throw nomad_output_partial_error("trinary_prod_cubes");
    }

    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
