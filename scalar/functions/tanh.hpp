#ifndef nomad__scalar__functions__tanh_hpp
#define nomad__scalar__functions__tanh_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double tanh(double x) { return std::tanh(x); }
  
  template <short autodiff_order>
  inline var<autodiff_order> tanh(const var<autodiff_order>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double t = std::tanh(input.first_val());
    
    push_dual_numbers<autodiff_order>(t);
    
    push_inputs(input.dual_numbers());
    
    double sech2 = 1.0 / std::cosh(input.first_val());
    sech2 *= sech2;
    
    if (autodiff_order >= 1) push_partials(sech2);
    if (autodiff_order >= 2) push_partials(-2 * sech2 * t);
    if (autodiff_order >= 3) push_partials(-2 * sech2 * sech2 + 4 * sech2 * t * t);

    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}

#endif
