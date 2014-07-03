#ifndef nomad__scalar__functions__fma_hpp
#define nomad__scalar__functions__fma_hpp

#include <math.h>
#include <var/var.hpp>

namespace nomad {
  
  inline double fma(double x, double y, double z) {
    return std::fma(x, y, z);
  }
  
  template <short autodiff_order>
  inline var<autodiff_order> fma(const var<autodiff_order>& v1,
                                 const var<autodiff_order>& v2,
                                 const var<autodiff_order>& v3) {
    
    const short partials_order = 2;
    const unsigned int n_inputs = 3;

    next_inputs_delta = n_inputs;
    next_partials_delta =
      var_body<autodiff_order, partials_order>::n_partials(n_inputs);
    
    new var_body<autodiff_order, partials_order>(n_inputs);
    
    double x = v1.first_val();
    double y = v2.first_val();
    double z = v3.first_val();
    
    push_dual_numbers<autodiff_order>(fma(x, y, z));
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    push_inputs(v3.dual_numbers());
    
    if (autodiff_order >= 1) {
      push_partials(y);
      push_partials(x);
      push_partials(1);
    }
    if (autodiff_order >= 2) {
      push_partials(0);
      push_partials(1);
      push_partials(0);
      
      push_partials(0);
      push_partials(0);
      push_partials(0);
    }

    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}

#endif
