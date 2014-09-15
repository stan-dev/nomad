#ifndef nomad__src__scalar__functions__smooth_functions__square_hpp
#define nomad__src__scalar__functions__smooth_functions__square_hpp

#include <src/var/var.hpp>
#include <src/var/derived/square_var_node.hpp>

namespace nomad {

  inline double square(double input) {
    return input * input;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    square(const var<AutodiffOrder, StrictSmoothness>& input) {
    
    create_node<square_var_node<AutodiffOrder>>(1);
    
    double val = input.first_val();

    push_dual_numbers<AutodiffOrder>(val * val);
    push_inputs(input.dual_numbers());
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
