#ifndef nomad__src__scalar__functions__nonsmooth_functions__fabs_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__fabs_hpp

#include <math.h>
#include <type_traits>

#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>

namespace nomad {
  
  inline double fabs(double x) { return std::fabs(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness> >::type
    fabs(const var<AutodiffOrder, StrictSmoothness>& input) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double x = input.first_val();
    push_dual_numbers<AutodiffOrder>(fabs(x));
    
    push_inputs(input.dual_numbers());
    
    if (AutodiffOrder >= 1) {
      if (x < 0)
        push_partials(-1);
      else
        push_partials(1);
    }

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
