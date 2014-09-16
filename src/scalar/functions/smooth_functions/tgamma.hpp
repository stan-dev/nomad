#ifndef nomad__src__scalar__functions__smooth_functions__tgamma_hpp
#define nomad__src__scalar__functions__smooth_functions__tgamma_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/scalar/functions/smooth_functions/polygamma.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double tgamma(double x) { return std::tgamma(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    tgamma(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "tgamma");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double val = input.first_val();
    double g = tgamma(val);
    
    push_dual_numbers<AutodiffOrder>(g);
    
    push_inputs(input.dual_numbers());
    
    if (AutodiffOrder >= 1) {
      double dg = digamma(val);
      push_partials(g * dg);
    
      if (AutodiffOrder >= 2) {
        double tg = trigamma(val);
        push_partials(g * (dg * dg + tg));
    
        if (AutodiffOrder >= 3)
          push_partials(g * (dg * dg * dg + 3.0 * dg * tg + quadrigamma(val)));

      }
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
