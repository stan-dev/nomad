#ifndef nomad__scalar__functions__acosh_hpp
#define nomad__scalar__functions__acosh_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double acosh(double x) { return std::acosh(x); }
  
  template <short autodiff_order>
  inline var<autodiff_order> acosh(const var<autodiff_order>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    const double x = input.first_val();
    push_dual_numbers<autodiff_order>(std::acosh(x));
    
    push_inputs(input.dual_numbers());
    
    double d1 = 1.0 / (x * x - 1);
    double d2 = std::sqrt(d1);
    
    if (autodiff_order >= 1) push_partials(d2);
    if (autodiff_order >= 2) push_partials(-x * d1 * d2);
    if (autodiff_order >= 3) push_partials((1 + 2 * x * x) * d1 * d1 * d2);

    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}

#endif
