#ifndef nomad__src__scalar__functions__smooth_functions__atanh_hpp
#define nomad__src__scalar__functions__smooth_functions__atanh_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double atanh(double x) { return std::atanh(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    atanh(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) {
      double val = input.first_val();
      validate_input(val, "atanh");
      validate_lower_bound(val, -1, "atanh");
      validate_upper_bound(val, 1, "atanh");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    const double x = input.first_val();
    push_dual_numbers<AutodiffOrder>(std::atanh(x));
    
    push_inputs(input.dual_numbers());
    
    double d1 = 1.0 / (1 - x * x);
    
    if (AutodiffOrder >= 1) push_partials(d1);
    if (AutodiffOrder >= 2) push_partials(2 * x * d1 * d1);
    if (AutodiffOrder >= 3) push_partials((2 + 6 * x * x) * d1 * d1 * d1);

    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_body_idx_ - 1);
    
  }

}

#endif
