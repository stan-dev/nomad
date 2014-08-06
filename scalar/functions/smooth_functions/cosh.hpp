#ifndef nomad__scalar__functions__smooth_functions__cosh_hpp
#define nomad__scalar__functions__smooth_functions__cosh_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double cosh(double x) { return std::cosh(x); }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    cosh(const var<autodiff_order, strict_smoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double c = std::cosh(input.first_val());
    double s = std::sinh(input.first_val());
    
    push_dual_numbers<autodiff_order>(c);
    
    push_inputs(input.dual_numbers());
    
    if (autodiff_order >= 1) push_partials(s);
    if (autodiff_order >= 2) push_partials(c);
    if (autodiff_order >= 3) push_partials(s);

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
