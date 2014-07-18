#ifndef nomad__scalar__functions__lgamma_hpp
#define nomad__scalar__functions__lgamma_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>
#include <scalar/functions/polygamma.hpp>

namespace nomad {
  
  inline double lgamma(double x) { return std::lgamma(x); }
  
  template <short autodiff_order>
  inline var<autodiff_order> lgamma(const var<autodiff_order>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double val = input.first_val();
    
    push_dual_numbers<autodiff_order>(lgamma(val));
    
    push_inputs(input.dual_numbers());
    
    if (autodiff_order >= 1) push_partials(digamma(val));
    if (autodiff_order >= 2) push_partials(trigamma(val));
    if (autodiff_order >= 3) push_partials(quadrigamma(val));

    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}

#endif
