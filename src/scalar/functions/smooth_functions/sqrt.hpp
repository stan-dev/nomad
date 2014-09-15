#ifndef nomad__src__scalar__functions__smooth_functions__sqrt_hpp
#define nomad__src__scalar__functions__smooth_functions__sqrt_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_body.hpp>
#include <src/autodiff/exceptions.hpp>

namespace nomad {
  
  inline double sqrt(double x) {
    return std::sqrt(x);
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    sqrt(const var<autodiff_order, strict_smoothness>& input) {
    
    //double input_val = input.first_val();
    //if (unlikely(std::isnan(input_val))) throw nomad_input_error("sqrt");
    //if (unlikely(input_val < 0)) throw nomad_domain_error("sqrt");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double val = std::sqrt(input.first_val());
    
    try {
      push_dual_numbers<autodiff_order>(val);
    } catch(nomad_error& e) {
      throw nomad_output_value_error("sqrt");
    }
    
    push_inputs(input.dual_numbers());
    
    double d2 = 1.0 / input.first_val();
    
    try {
      if (autodiff_order >= 1) push_partials(val *= 0.5 * d2);
      if (autodiff_order >= 2) push_partials(val *= - 0.5 * d2);
      if (autodiff_order >= 3) push_partials(val *= - 1.5 * d2);
    } catch(nomad_error& e) {
      throw nomad_output_partial_error("sqrt");
    }

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
