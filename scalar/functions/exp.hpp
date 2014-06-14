#ifndef nomad__scalar__functions__exp_hpp
#define nomad__scalar__functions__exp_hpp

#include <math.h>
#include <var/var.hpp>

namespace nomad {
  
  template <short autodiff_order>
  inline var<autodiff_order> exp(const var<autodiff_order>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      var_body<autodiff_order, partials_order>::n_partials(n_inputs);
    
    new var_body<autodiff_order, partials_order>(n_inputs);

    double val = std::exp(input.first_val());
    
    push_dual_numbers<autodiff_order>(val);
    
    push_inputs(input.dual_numbers());
    
    if (autodiff_order >= 1) push_partials(val);
    if (autodiff_order >= 2) push_partials(val);
    if (autodiff_order >= 3) push_partials(val);

    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}

#endif
