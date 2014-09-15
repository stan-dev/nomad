#ifndef nomad__src__scalar__functions__smooth_functions__hypot_hpp
#define nomad__src__scalar__functions__smooth_functions__hypot_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/var/derived/binary_var_node.hpp>
#include <src/autodiff/validation.hpp>

namespace nomad {
  
  inline double hypot(double x, double y) {
    return std::hypot(x, y);
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    hypot(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
          const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    
    if (ValidateIO) {
      validate_input(v1.first_val(), "hypot");
      validate_input(v2.first_val(), "hypot");
    }
      
    const short partials_order = 3;
    const unsigned int n_inputs = 2;
    
    create_node<binary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    double x = v1.first_val();
    double y = v2.first_val();
    
    if (x > y) {
      
      double r = y / x;
      
      double val = x * sqrt(1 + r * r);
      push_dual_numbers<AutodiffOrder>(val);
      
      double d1 = 1.0 / val;
      
      if (AutodiffOrder >= 1) {
        push_partials(x * d1);
        push_partials(y * d1);
      }
      if (AutodiffOrder >= 2) {
        d1 /= 1.0 + r * r;
        
        push_partials(r * r * d1);
        push_partials(-r * d1);
        push_partials(d1);
      }
      if (AutodiffOrder >= 3) {
        d1 /= -(1.0 + r * r);
        double inv_x = 1.0 / x;
          
        push_partials(3.0 * r * r * inv_x * d1);
        push_partials(r * inv_x * (r * r - 2.0) * d1);
        push_partials(inv_x * (1.0 - 2 * r * r) * d1);
        push_partials(3 * r * inv_x * d1);
      }
      
    } else {
        
      double r = x / y;
      
      double val = y * sqrt(1 + r * r);
      push_dual_numbers<AutodiffOrder>(val);
      
      double d1 = 1.0 / val;
      
      if (AutodiffOrder >= 1) {
        push_partials(x * d1);
        push_partials(y * d1);
      }
      if (AutodiffOrder >= 2) {
        d1 /= 1.0 + r * r;
        
        push_partials(d1);
        push_partials(-r * d1);
        push_partials(r * r * d1);
      }
      if (AutodiffOrder >= 3) {
        d1 /= -(1.0 + r * r);
        double inv_y = 1.0 / y;
        
        push_partials(3.0 * r * inv_y * d1);
        push_partials(inv_y * (1.0 - 2 * r * r) * d1);
        push_partials(r * inv_y * (r * r - 2.0) * d1);
        push_partials(3 * r * r * inv_y * d1);
      }
      
    }

    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_body_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    hypot(double x,
          const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v2) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    push_inputs(v2.dual_numbers());
    
    double y = v2.first_val();
    
    if (x > y) {
      
      double r = y / x;
      
      double val = x * sqrt(1 + r * r);
      push_dual_numbers<AutodiffOrder>(val);
      
      double d1 = 1.0 / val;
      
      if (AutodiffOrder >= 1) {
        push_partials(y * d1);
      }
      if (AutodiffOrder >= 2) {
        d1 /= 1.0 + r * r;
        push_partials(d1);
      }
      if (AutodiffOrder >= 3) {
        d1 /= -(1.0 + r * r);
        push_partials(3 * r * d1 / x);
      }
      
    } else {
      
      double r = x / y;
      
      double val = y * sqrt(1 + r * r);
      push_dual_numbers<AutodiffOrder>(val);
      
      double d1 = 1.0 / val;
      
      if (AutodiffOrder >= 1) {
        push_partials(y * d1);
      }
      if (AutodiffOrder >= 2) {
        d1 /= 1.0 + r * r;
        push_partials(r * r * d1);
      }
      if (AutodiffOrder >= 3) {
        d1 /= -(1.0 + r * r);
        push_partials(3 * r * r * d1 / y);
      }
      
    }
    
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_body_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness, bool ValidateIO>
  inline var<AutodiffOrder, StrictSmoothness, ValidateIO>
    hypot(const var<AutodiffOrder, StrictSmoothness, ValidateIO>& v1,
          double y) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    push_inputs(v1.dual_numbers());
    
    double x = v1.first_val();
    
    if (x > y) {
      
      double r = y / x;
      
      double val = x * sqrt(1 + r * r);
      push_dual_numbers<AutodiffOrder>(val);

      double d1 = 1.0 / val;
      
      if (AutodiffOrder >= 1) {
        push_partials(x * d1);
      }
      if (AutodiffOrder >= 2) {
        d1 /= 1.0 + r * r;
        push_partials(r * r * d1);
      }
      if (AutodiffOrder >= 3) {
        d1 /= -(1.0 + r * r);
        push_partials(3.0 * r * r * d1 / x);
      }
      
    } else {
      
      double r = x / y;
      
      double val = y * sqrt(1 + r * r);
      push_dual_numbers<AutodiffOrder>(val);
      
      double d1 = 1.0 / val;
      
      if (AutodiffOrder >= 1) {
        push_partials(x * d1);
      }
      if (AutodiffOrder >= 2) {
        d1 /= 1.0 + r * r;
        push_partials(d1);
      }
      if (AutodiffOrder >= 3) {
        d1 /= -(1.0 + r * r);
        push_partials(3.0 * r * d1 / y);
      }
      
    }
    
    return var<AutodiffOrder, StrictSmoothness, ValidateIO>(next_body_idx_ - 1);
    
  }

}

#endif
