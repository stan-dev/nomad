#ifndef nomad__src__scalar__operators__smooth_operators__operator_unary_plus_hpp
#define nomad__src__scalar__operators__smooth_operators__operator_unary_plus_hpp

#include <src/var/var.hpp>
#include <src/var/derived/unary_plus_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    operator+(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1) {

    if (ValidateIO) validate_input(v1.first_val(), "operator+");
      
    create_node<unary_plus_var_node<AutodiffOrder>>(1);
    
    push_dual_numbers<AutodiffOrder>(v1.first_val());
    
    push_inputs(v1.dual_numbers());
    
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_body_idx_ - 1);
    
  }

}
#endif
