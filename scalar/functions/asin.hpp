#ifndef nomad__scalar__functions__asin_hpp
#define nomad__scalar__functions__asin_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>

namespace nomad {

  inline double asin(double x) { return std::asin(x); }
  
  template <short autodiff_order>
  inline var<autodiff_order> asin(const var<autodiff_order>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    const double x = input.first_val();
    push_dual_numbers<autodiff_order>(std::asin(x));
    
    push_inputs(input.dual_numbers());
    
    const double d1 = 1.0 / (1 - x * x);
    const double d2 = std::sqrt(d1);
    
    if (autodiff_order >= 1) push_partials(d2);
    if (autodiff_order >= 2) push_partials(x * d1 * d2);
    if (autodiff_order >= 3) push_partials((1 + 2 * x * x) * d1 * d1 * d2);

    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}

#endif
