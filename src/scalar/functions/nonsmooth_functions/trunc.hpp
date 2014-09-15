#ifndef nomad__src__scalar__functions__nonsmooth_functions__trunc_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__trunc_hpp

#include <math.h>
#include <type_traits>

#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>

namespace nomad {
  
  inline double trunc(double x) { return std::trunc(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness> >::type
    trunc(const var<AutodiffOrder, StrictSmoothness>& input) {
    
    const short partials_order = 0;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_node<AutodiffOrder, partials_order>::n_partials();
    
    new unary_var_node<AutodiffOrder, partials_order>();

    push_dual_numbers<AutodiffOrder>(trunc(input.first_val()));
    
    push_inputs(input.dual_numbers());

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
