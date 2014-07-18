#ifndef nomad__scalar__functions__tgamma_hpp
#define nomad__scalar__functions__tgamma_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>
#include <scalar/functions/polygamma.hpp>

namespace nomad {
  
  inline double tgamma(double x) { return std::tgamma(x); }
  
  template <short autodiff_order>
  inline var<autodiff_order> tgamma(const var<autodiff_order>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double val = input.first_val();
    double g = tgamma(val);
    
    push_dual_numbers<autodiff_order>(g);
    
    push_inputs(input.dual_numbers());
    
    if (autodiff_order >= 1) {
      double dg = digamma(val);
      push_partials(g * dg);
    
      if (autodiff_order >= 2) {
        double tg = trigamma(val);
        push_partials(g * (dg * dg + tg));
    
        if (autodiff_order >= 3)
          push_partials(g * (dg * dg * dg + 3.0 * dg * tg + quadrigamma(val)));

      }
    }
      
    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}

#endif
