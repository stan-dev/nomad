#ifndef nomad__src__scalar__functions__smooth_functions__atan_hpp
#define nomad__src__scalar__functions__smooth_functions__atan_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>

namespace nomad {

  inline double atan(double x) { return std::atan(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    atan(const var<AutodiffOrder, StrictSmoothness>& input) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    const double x = input.first_val();
    push_dual_numbers<AutodiffOrder>(std::atan(x));
    
    push_inputs(input.dual_numbers());
    
    const double d1 = 1.0 / (1 + x * x);
    
    if (AutodiffOrder >= 1) push_partials(d1);
    if (AutodiffOrder >= 2) push_partials(- 2 * x * d1 * d1);
    if (AutodiffOrder >= 3) push_partials((- 2 + 6 * x * x) * d1 * d1 * d1);

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
