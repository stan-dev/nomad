#ifndef nomad__scalar__functions__Phi_hpp
#define nomad__scalar__functions__Phi_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double Phi(double x) {
    if (x < -40)
      return 0;
    else if (x < 0)
      return 0.5 * std::erfc(-0.70710678118655 * x);
    else if (x < 40)
      return 0.5 * (1.0 + std::erf(0.70710678118655 * x));
    else
      return 1;
  }
  
  template <short autodiff_order>
  inline var<autodiff_order> Phi(const var<autodiff_order>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double x = input.first_val();
    
    push_dual_numbers<autodiff_order>(erf(x));
    
    push_inputs(input.dual_numbers());
    
    double C = 0.39894228040143 * exp(- x * x);
    
    if (autodiff_order >= 1) push_partials(C);
    if (autodiff_order >= 2) push_partials(- 2 * x * C);
    if (autodiff_order >= 3) push_partials(2 * (2 * x * x - 1) * C);

    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}

#endif
