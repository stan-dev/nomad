#ifndef nomad__src__scalar__functions__smooth_functions__square_hpp
#define nomad__src__scalar__functions__smooth_functions__square_hpp

#include <src/var/var.hpp>
#include <src/var/derived/square_var_body.hpp>

namespace nomad {

  inline double square(double input) {
    return input * input;
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    square(const var<autodiff_order, strict_smoothness>& input) {
    
    next_inputs_delta = 1;
    // next_partials_delta not used by square_var_body
    
    new square_var_body<autodiff_order>();
    
    double val = input.first_val();

    push_dual_numbers<autodiff_order>(val * val);
    push_inputs(input.dual_numbers());
    
    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
