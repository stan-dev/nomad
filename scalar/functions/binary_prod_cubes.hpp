#ifndef nomad__scalar__functions__binary_prod_cubes_hpp
#define nomad__scalar__functions__binary_prod_cubes_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/binary_var_body.hpp>

namespace nomad {
  
  inline double binary_prod_cubes(double x, double y) {
    return x * x * x * y * y * y;
  }
  
  template <short autodiff_order>
  inline var<autodiff_order> binary_prod_cubes(const var<autodiff_order>& v1,
                                               const var<autodiff_order>& v2) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 2;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      binary_var_body<autodiff_order, partials_order>::n_partials();
    
    new binary_var_body<autodiff_order, partials_order>();

    double x = v1.first_val();
    double y = v2.first_val();
    
    push_dual_numbers<autodiff_order>(binary_prod_cubes(x, y));
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    if (autodiff_order >= 1) {
      push_partials(3 * x * x * y * y * y);
      push_partials(3 * x * x * x * y * y);
    }
    if (autodiff_order >= 2) {
      push_partials(6 * x * y * y * y);
      push_partials(9 * x * x * y * y);
      push_partials(6 * x * x * x * y);
    }
    if (autodiff_order >= 3) {
      push_partials(6 * y * y * y);
      push_partials(18 * x * y * y);
      push_partials(18 * x * x * y);
      push_partials(6 * x * x * x);
    }

    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}

#endif
