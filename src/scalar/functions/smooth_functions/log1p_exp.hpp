#ifndef nomad__src__scalar__functions__smooth_functions__log1p_exp_hpp
#define nomad__src__scalar__functions__smooth_functions__log1p_exp_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double log1p_exp(double x) {
    if (x > 0) return x + log1p(exp(-x));
    return std::log1p(exp(x));
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    log1p_exp(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& input) {
    
    if (ValidateIO) validate_input(input.first_val(), "log1p_exp");
      
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double val = input.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(log1p_exp(val));
    } catch (nomad_error) {
      throw nomad_output_value_error("log1p_exp");
    }
      
    push_inputs(input.dual_numbers());
    
    try {
      if (val > 0) {
        double e = exp(-val);
        double p = 1.0 / (1.0 + e);
        if (AutodiffOrder >= 1) push_partials<ValidateIO>(p);
        if (AutodiffOrder >= 2) push_partials<ValidateIO>(p * p * e);
        if (AutodiffOrder >= 3) push_partials<ValidateIO>(p * (2.0 * p * p - 3.0 * p + 1.0));
      } else {
        double e = exp(val);
        double p = e / (1.0 + e);
        if (AutodiffOrder >= 1) push_partials<ValidateIO>(p);
        if (AutodiffOrder >= 2) push_partials<ValidateIO>(p * p / e);
        if (AutodiffOrder >= 3) push_partials<ValidateIO>(p * (2.0 * p * p - 3.0 * p + 1.0));
      }
    } catch (nomad_error) {
      throw nomad_output_partial_error("log1p_exp");
    }

    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
