#ifndef nomad__src__scalar__functions__smooth_functions__exp2_hpp
#define nomad__src__scalar__functions__smooth_functions__exp2_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double exp2(double x) { return std::exp2(x); }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    exp2(const var<autodiff_order, strict_smoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double val = exp2(input.first_val());
    
    push_dual_numbers<autodiff_order>(val);
    
    push_inputs(input.dual_numbers());
    
    double log2 = 0.693147180559945;
    
    if (autodiff_order >= 1) push_partials(val *= log2);
    if (autodiff_order >= 2) push_partials(val *= log2);
    if (autodiff_order >= 3) push_partials(val *= log2);

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
