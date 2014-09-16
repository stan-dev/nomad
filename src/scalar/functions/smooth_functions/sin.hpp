#ifndef nomad__src__scalar__functions__smooth_functions__sin_hpp
#define nomad__src__scalar__functions__smooth_functions__sin_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double sin(double x) { return std::sin(x); }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    sin(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "sin");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double c = std::cos(input.first_val());
    double s = std::sin(input.first_val());
    
    push_dual_numbers<AutodiffOrder>(s);
    
    push_inputs(input.dual_numbers());
    
    if (AutodiffOrder >= 1) push_partials(c);
    if (AutodiffOrder >= 2) push_partials(-s);
    if (AutodiffOrder >= 3) push_partials(-c);

    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
