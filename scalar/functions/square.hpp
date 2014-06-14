#ifndef nomad__scalar__functions__square_hpp
#define nomad__scalar__functions__square_hpp

#include <var/var.hpp>
#include <var/derived/square_var_body.hpp>

namespace nomad {

  inline double square(double input) {
    return input * input;
  }
  
  template <short autodiff_order>
  inline var<autodiff_order> square(const var<autodiff_order>& input) {
    
    next_inputs_delta = 1;
    // next_partials_delta not used by square_var_body
    
    new square_var_body<autodiff_order>();
    
    double val = input.first_val();

    push_dual_numbers<autodiff_order>(val * val);
    push_inputs(input.dual_numbers());
    
    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}

#endif
