#ifndef nomad__src__scalar__functions__nonsmooth_functions__fdim_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__fdim_hpp

#include <math.h>
#include <type_traits>

#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/var/derived/binary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double fdim(double x, double y) { return std::fdim(x, y); }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness, ValidateIO> >::type
    fdim(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
         const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    
    if (ValidateIO) {
      validate_input(v1.first_val(), "fdim");
      validate_input(v2.first_val(), "fdim");
    }
    
    const short partials_order = 1;
    const unsigned int n_inputs = 2;
    
    create_node<binary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v1.first_val();
    double y = v2.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(fdim(x, y));
    } catch (nomad_error) {
      throw nomad_output_value_error("fdim");
    }
      
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    try {
      if (AutodiffOrder >= 1) {
        if (x > y) {
          push_partials<ValidateIO>(1);
          push_partials<ValidateIO>(-1);
        }
        else {
          push_partials<ValidateIO>(0);
          push_partials<ValidateIO>(0);
        }
      }
    } catch (nomad_error) {
      throw nomad_output_partial_error("fdim");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness, ValidateIO> >::type
    fdim(double x,
         const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    
    if (ValidateIO) {
      validate_input(x, "fdim");
      validate_input(v2.first_val(), "fdim");
    }
      
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double y = v2.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(fdim(x, y));
    } catch (nomad_error) {
      throw nomad_output_value_error("fdim");
    }
      
    push_inputs(v2.dual_numbers());
    
    try {
      if (AutodiffOrder >= 1) {
        if (x > y) push_partials<ValidateIO>(-1);
        else       push_partials<ValidateIO>(0);
      }
    } catch (nomad_error) {
      throw nomad_output_partial_error("fdim");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness, ValidateIO> >::type
    fdim(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
         double y) {
    
    if (ValidateIO) {
      validate_input(v1.first_val(), "fdim");
      validate_input(y, "fdim");
    }
      
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v1.first_val();
    
    try {
      push_dual_numbers<AutodiffOrder, ValidateIO>(fdim(x, y));
    } catch (nomad_error) {
      throw nomad_output_value_error("fdim");
    }
      
    push_inputs(v1.dual_numbers());
    
    try {
      if (AutodiffOrder >= 1) {
        if (x > y) push_partials<ValidateIO>(1);
        else       push_partials<ValidateIO>(0);
      }
    } catch (nomad_error) {
      throw nomad_output_partial_error("fdim");
    }
      
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_node_idx_ - 1);
    
  }

}

#endif
