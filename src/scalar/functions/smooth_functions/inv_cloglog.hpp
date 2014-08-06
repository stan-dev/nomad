#ifndef nomad__src__scalar__functions__smooth_functions__inv_cloglog_hpp
#define nomad__src__scalar__functions__smooth_functions__inv_cloglog_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_body.hpp>

namespace nomad {
  
  inline double inv_cloglog(double x) { return 1.0 - std::exp(-std::exp(x)); }
  
  template <short autodiff_order, bool strict_smoothness>
  inline var<autodiff_order, strict_smoothness>
    inv_cloglog(const var<autodiff_order, strict_smoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();

    double e = std::exp(input.first_val());
    double ee = std::exp(-e);
    
    push_dual_numbers<autodiff_order>(1 - ee);
    
    push_inputs(input.dual_numbers());
    
    if (autodiff_order >= 1) push_partials(ee * e);
    if (autodiff_order >= 2) push_partials(- ee * e * (e - 1));
    if (autodiff_order >= 3) push_partials(ee * e * (1 + e * (e - 3)) );

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
