#ifndef nomad__scalar__functions__tan_hpp
#define nomad__scalar__functions__tan_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double tan(double x) { return std::tan(x); }
  
  template <short autodiff_order>
  inline var<autodiff_order> tan(const var<autodiff_order>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double t = std::tan(input.first_val());
    
    push_dual_numbers<autodiff_order>(t);
    
    push_inputs(input.dual_numbers());
    
    double sec2 = 1.0 / std::cos(input.first_val());
    sec2 *= sec2;
    
    if (autodiff_order >= 1) push_partials(sec2);
    if (autodiff_order >= 2) push_partials(2 * sec2 * t);
    if (autodiff_order >= 3) push_partials(2 * sec2 * sec2 + 4 * sec2 * t * t);

    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}

#endif
