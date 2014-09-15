#ifndef nomad__src__scalar__functions__smooth_functions__erfc_hpp
#define nomad__src__scalar__functions__smooth_functions__erfc_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>

namespace nomad {
  
  inline double erfc(double x) { return std::erfc(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    erfc(const var<AutodiffOrder, StrictSmoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double x = input.first_val();
    
    push_dual_numbers<AutodiffOrder>(erfc(x));
    
    push_inputs(input.dual_numbers());
    
    double C = - 2 * 0.56418958354776 * exp(- x * x);
    
    if (AutodiffOrder >= 1) push_partials(C);
    if (AutodiffOrder >= 2) push_partials(- 2 * x * C);
    if (AutodiffOrder >= 3) push_partials(2 * (2 * x * x - 1) * C);

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
