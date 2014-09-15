#ifndef nomad__src__scalar__functions__smooth_functions__log_sum_exp_hpp
#define nomad__src__scalar__functions__smooth_functions__log_sum_exp_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/var/derived/binary_var_node.hpp>

namespace nomad {
  
  inline double log_sum_exp(double x, double y) {
    if (x > y)
      return x + std::log(std::exp(y - x) + 1);
    else
      return y + std::log(std::exp(x - y) + 1);
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    log_sum_exp(const var<AutodiffOrder, StrictSmoothness>& v1,
                const var<AutodiffOrder, StrictSmoothness>& v2) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 2;
      
    create_node<binary_var_node<AutodiffOrder, partials_order>>(n_inputs);

    double x = v1.first_val();
    double y = v2.first_val();
    
    push_dual_numbers<AutodiffOrder>(log_sum_exp(x, y));
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    if (x > y) {
    
      double e = std::exp(y - x);
      double p = e / (1.0 + e);
      
      if (AutodiffOrder >= 1) {
        push_partials(1.0 / (1.0 + e));
        push_partials(p);
      }
      if (AutodiffOrder >= 2) {
        double p2 = p * p / e;
        push_partials(p2);
        push_partials(-p2);
        push_partials(p2);
      }
      if (AutodiffOrder >= 3) {
        p *= 2 * p * p - 3 * p + 1;
        push_partials(-p);
        push_partials(p);
        push_partials(-p);
        push_partials(p);
      }
      
    } else {
      
      double e = std::exp(x - y);
      double p = e / (1.0 + e);
      
      if (AutodiffOrder >= 1) {
        push_partials(p);
        push_partials(1.0 / (1.0 + e));
      }
      if (AutodiffOrder >= 2) {
        double p2 = p * p / e;
        push_partials(p2);
        push_partials(-p2);
        push_partials(p2);
      }
      if (AutodiffOrder >= 3) {
        p *= 2 * p * p - 3 * p + 1;
        push_partials(p);
        push_partials(-p);
        push_partials(p);
        push_partials(-p);
      }
      
    }

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    log_sum_exp(double x,
                const var<AutodiffOrder, StrictSmoothness>& v2) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double y = v2.first_val();
    push_dual_numbers<AutodiffOrder>(log_sum_exp(x, y));
    
    push_inputs(v2.dual_numbers());
    
    if (x > y) {
      
      double e = std::exp(y - x);
      double p = e / (1.0 + e);
      
      if (AutodiffOrder >= 1) {
        push_partials(p);
      }
      if (AutodiffOrder >= 2) {
        push_partials(p * p / e);
      }
      if (AutodiffOrder >= 3) {
        p *= 2 * p * p - 3 * p + 1;
        push_partials(p);
      }
      
    } else {
      
      double e = std::exp(x - y);
      double p = 1.0 / (1.0 + e);
      
      if (AutodiffOrder >= 1) {
        push_partials(p);
      }
      if (AutodiffOrder >= 2) {
        p *= e;
        push_partials(p * p / e);
      }
      if (AutodiffOrder >= 3) {
        p *= 2 * p * p - 3 * p + 1;
        push_partials(-p);
      }
      
    }
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    log_sum_exp(const var<AutodiffOrder, StrictSmoothness>& v1,
                double y) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v1.first_val();
    push_dual_numbers<AutodiffOrder>(log_sum_exp(x, y));
    
    push_inputs(v1.dual_numbers());
    
    if (x > y) {
      
      double e = std::exp(y - x);
      double p = 1.0 / (1.0 + e);
      
      if (AutodiffOrder >= 1) {
        push_partials(p);
      }
      if (AutodiffOrder >= 2) {
        p *= e;
        push_partials(p * p / e);
      }
      if (AutodiffOrder >= 3) {
        p *= 2 * p * p - 3 * p + 1;
        push_partials(-p);
      }
      
    } else {
      
      double e = std::exp(x - y);
      double p = e / (1.0 + e);
      
      if (AutodiffOrder >= 1) {
        push_partials(p);
      }
      if (AutodiffOrder >= 2) {
        push_partials(p * p / e);
      }
      if (AutodiffOrder >= 3) {
        p *= 2 * p * p - 3 * p + 1;
        push_partials(p);
      }
      
    }
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
