#ifndef nomad__scalar__functions__nonsmooth_functions__round_hpp
#define nomad__scalar__functions__nonsmooth_functions__round_hpp

#include <math.h>
#include <type_traits>

#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double round(double x) { return std::round(x); }
  
  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, var<autodiff_order, strict_smoothness> >::type
    round(const var<autodiff_order, strict_smoothness>& input) {
    
    const short partials_order = 0;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    push_dual_numbers<autodiff_order>(round(input.first_val()));
    
    push_inputs(input.dual_numbers());

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
