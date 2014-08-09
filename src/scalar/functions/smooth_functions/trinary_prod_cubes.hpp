#ifndef nomad__src__scalar__functions__smooth_functions__trinary_prod_cubes_hpp
#define nomad__src__scalar__functions__smooth_functions__trinary_prod_cubes_hpp

#include <src/var/var.hpp>

namespace nomad {
  
  inline double trinary_prod_cubes(double x, double y, double z) {
    return x * x * x * y * y * y * z * z * z;
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    trinary_prod_cubes(const var<autodiff_order, strict_smoothness>& v1,
                       const var<autodiff_order, strict_smoothness>& v2,
                       const var<autodiff_order, strict_smoothness>& v3) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 3;

    next_inputs_delta = n_inputs;
    next_partials_delta =
      var_body<autodiff_order, partials_order>::n_partials(n_inputs);
    
    new var_body<autodiff_order, partials_order>(n_inputs);
    
    double x = v1.first_val();
    double y = v2.first_val();
    double z = v3.first_val();
    
    push_dual_numbers<autodiff_order>(trinary_prod_cubes(x, y, z));
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    push_inputs(v3.dual_numbers());
    
    if (autodiff_order >= 1) {
      push_partials(3 * x * x * y * y * y * z * z * z);
      push_partials(3 * x * x * x * y * y * z * z * z);
      push_partials(3 * x * x * x * y * y * y * z * z);
    }
    if (autodiff_order >= 2) {
      push_partials(6 * x * y * y * y * z * z * z);
      push_partials(9 * x * x * y * y * z * z * z);
      push_partials(6 * x * x * x * y * z * z * z);
      
      push_partials(9 * x * x * y * y * y * z * z);
      push_partials(9 * x * x * x * y * y * z * z);
      push_partials(6 * x * x * x * y * y * y * z);
    }
    if (autodiff_order >= 3) {
      push_partials(6 * y * y * y * z * z * z);
      push_partials(18 * x * y * y * z * z * z);
      push_partials(18 * x * x * y * z * z * z);
      push_partials(6 * x * x * x * z * z * z);
      push_partials(18 * x * y * y * y * z * z);
      push_partials(27 * x * x * y * y * z * z);
      push_partials(18 * x * x * x * y * z * z);
      push_partials(18 * x * x * y * y * y * z);
      push_partials(18 * x * x * x * y * y * z);
      push_partials(6 * x * x * x * y * y * y);
    }

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
