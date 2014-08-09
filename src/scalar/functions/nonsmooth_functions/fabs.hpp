#ifndef nomad__src__scalar__functions__nonsmooth_functions__fabs_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__fabs_hpp

#include <math.h>
#include <type_traits>

#include <src/var/var.hpp>
#include <src/var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double fabs(double x) { return std::fabs(x); }
  
  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, var<autodiff_order, strict_smoothness> >::type
    fabs(const var<autodiff_order, strict_smoothness>& input) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double x = input.first_val();
    push_dual_numbers<autodiff_order>(fabs(x));
    
    push_inputs(input.dual_numbers());
    
    if (autodiff_order >= 1) {
      if (x < 0)
        push_partials(-1);
      else
        push_partials(1);
    }

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
