#ifndef nomad__src__scalar__functions__nonsmooth_functions__fdim_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__fdim_hpp

#include <math.h>
#include <type_traits>

#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/var/derived/binary_var_node.hpp>

namespace nomad {
  
  inline double fdim(double x, double y) { return std::fdim(x, y); }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness> >::type
    fdim(const var<AutodiffOrder, StrictSmoothness>& v1,
         const var<AutodiffOrder, StrictSmoothness>& v2) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 2;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
    binary_var_node<AutodiffOrder, partials_order>::n_partials();
    
    new binary_var_node<AutodiffOrder, partials_order>();
    
    double x = v1.first_val();
    double y = v2.first_val();
    
    push_dual_numbers<AutodiffOrder>(fdim(x, y));
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    
    if (AutodiffOrder >= 1) {
      if (x > y) {
        push_partials(1);
        push_partials(-1);
      }
      else {
        push_partials(0);
        push_partials(0);
      }
    }
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness> >::type
    fdim(double x,
         const var<AutodiffOrder, StrictSmoothness>& v2) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
    unary_var_node<AutodiffOrder, partials_order>::n_partials();
    
    new unary_var_node<AutodiffOrder, partials_order>();
    
    double y = v2.first_val();
    
    push_dual_numbers<AutodiffOrder>(fdim(x, y));
    
    push_inputs(v2.dual_numbers());
    
    if (AutodiffOrder >= 1) {
      if (x > y) push_partials(-1);
      else       push_partials(0);
    }
      
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness> >::type
    fdim(const var<AutodiffOrder, StrictSmoothness>& v1,
         double y) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
    unary_var_node<AutodiffOrder, partials_order>::n_partials();
    
    new unary_var_node<AutodiffOrder, partials_order>();
    
    double x = v1.first_val();
    
    push_dual_numbers<AutodiffOrder>(fdim(x, y));
    
    push_inputs(v1.dual_numbers());
    
    if (AutodiffOrder >= 1) {
      if (x > y) push_partials(1);
      else       push_partials(0);
    }

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
