#ifndef nomad__src__scalar__functions__nonsmooth_functions__ceil_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__ceil_hpp

#include <math.h>
#include <type_traits>

#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>

namespace nomad {
  
  inline double ceil(double x) { return std::ceil(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness> >::type
    ceil(const var<AutodiffOrder, StrictSmoothness>& input) {
    
    const short partials_order = 0;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    push_dual_numbers<AutodiffOrder>(ceil(input.first_val()));
    
    push_inputs(input.dual_numbers());

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
