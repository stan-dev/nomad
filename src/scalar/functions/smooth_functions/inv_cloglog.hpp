#ifndef nomad__src__scalar__functions__smooth_functions__inv_cloglog_hpp
#define nomad__src__scalar__functions__smooth_functions__inv_cloglog_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double inv_cloglog(double x) { return 1.0 - std::exp(-std::exp(x)); }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    inv_cloglog(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "inv_cloglog");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double e = std::exp(input.first_val());
    double ee = std::exp(-e);
    
    push_dual_numbers<AutodiffOrder>(1 - ee);
    
    push_inputs(input.dual_numbers());
    
    if (AutodiffOrder >= 1) push_partials(ee * e);
    if (AutodiffOrder >= 2) push_partials(- ee * e * (e - 1));
    if (AutodiffOrder >= 3) push_partials(ee * e * (1 + e * (e - 3)) );

    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_body_idx_ - 1);
    
  }

}

#endif
