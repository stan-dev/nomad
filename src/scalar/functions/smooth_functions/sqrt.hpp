#ifndef nomad__src__scalar__functions__smooth_functions__sqrt_hpp
#define nomad__src__scalar__functions__smooth_functions__sqrt_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/exceptions.hpp>

namespace nomad {
  
  inline double sqrt(double x) {
    return std::sqrt(x);
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    sqrt(const var<AutodiffOrder, StrictSmoothness>& input) {
    
    //double input_val = input.first_val();
    //if (unlikely(std::isnan(input_val))) throw nomad_input_error("sqrt");
    //if (unlikely(input_val < 0)) throw nomad_domain_error("sqrt");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double val = std::sqrt(input.first_val());
    
    try {
      push_dual_numbers<AutodiffOrder>(val);
    } catch(nomad_error& e) {
      throw nomad_output_value_error("sqrt");
    }
    
    push_inputs(input.dual_numbers());
    
    double d2 = 1.0 / input.first_val();
    
    try {
      if (AutodiffOrder >= 1) push_partials(val *= 0.5 * d2);
      if (AutodiffOrder >= 2) push_partials(val *= - 0.5 * d2);
      if (AutodiffOrder >= 3) push_partials(val *= - 1.5 * d2);
    } catch(nomad_error& e) {
      throw nomad_output_partial_error("sqrt");
    }

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
