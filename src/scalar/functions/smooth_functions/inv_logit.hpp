#ifndef nomad__src__scalar__functions__smooth_functions__inv_logit_hpp
#define nomad__src__scalar__functions__smooth_functions__inv_logit_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double inv_logit(double x) {
    if (x > 0)
      return 1.0 / (1.0 + exp(-x));
    else {
      double e = std::exp(x);
      return e / (1.0 + e);
    }
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    inv_logit(const var<autodiff_order, strict_smoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double s = inv_logit(input.first_val());
    
    push_dual_numbers<autodiff_order>(s);
    
    push_inputs(input.dual_numbers());
    
    double ds = s * (1 - s);
    
    if (autodiff_order >= 1) push_partials(ds);
    if (autodiff_order >= 2) push_partials(ds * (1 - 2 * s) );
    if (autodiff_order >= 3) push_partials(ds * (1 - 6 * s * (1 - s)) );

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
