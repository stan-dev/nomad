#ifndef nomad__src__scalar__functions__smooth_functions__erf_hpp
#define nomad__src__scalar__functions__smooth_functions__erf_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double erf(double x) { return std::erf(x); }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    erf(const var<autodiff_order, strict_smoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double x = input.first_val();
    
    push_dual_numbers<autodiff_order>(erf(x));
    
    push_inputs(input.dual_numbers());
    
    double C = 2 * 0.56418958354776 * exp(- x * x);
    
    if (autodiff_order >= 1) push_partials(C);
    if (autodiff_order >= 2) push_partials(- 2 * x * C);
    if (autodiff_order >= 3) push_partials(2 * (2 * x * x - 1) * C);

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
