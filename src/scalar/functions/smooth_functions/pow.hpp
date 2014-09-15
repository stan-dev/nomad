#ifndef nomad__src__scalar__functions__smooth_functions__pow_hpp
#define nomad__src__scalar__functions__smooth_functions__pow_hpp

#include <math.h>
#include <src/var/var.hpp>
#include <src/var/derived/unary_var_node.hpp>
#include <src/var/derived/binary_var_node.hpp>

namespace nomad {
  
  inline double pow(double x, double y) { return std::pow(x, y); }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    pow(const var<AutodiffOrder, StrictSmoothness>& v1,
        const var<AutodiffOrder, StrictSmoothness>& v2) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 2;
    
    create_node<binary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v1.first_val();
    double y = v2.first_val();
    double val = std::pow(x, y);
    
    push_dual_numbers<AutodiffOrder>(val);
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    double lx = std::log(x);
    
    if (AutodiffOrder >= 1) {
      push_partials(val * y / x);
      push_partials(val * lx);
    }
    if (AutodiffOrder >= 2) {
      push_partials( y * (y - 1.0) * val / (x * x) );
      push_partials( (1 + y * lx) * val / x );
      push_partials( lx * lx * val );
    }
    if (AutodiffOrder >= 3) {
      push_partials((y - 2.0) * (y - 1.0) * y * val / (x  * x * x) );
      push_partials( ((y - 1.0) * y * lx + 2 * y - 1.0) * val / (x * x) );
      push_partials( lx * (2.0 + y * lx) * val / x );
      push_partials( lx * lx * lx * val );
    }
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    pow(double x,
        const var<AutodiffOrder, StrictSmoothness>& v2) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double y = v2.first_val();
    double val = std::pow(x, y);
    
    push_dual_numbers<AutodiffOrder>(val);
    
    push_inputs(v2.dual_numbers());
    
    double lx = std::log(x);
    
    if (AutodiffOrder >= 1) push_partials(val * lx);
    if (AutodiffOrder >= 2) push_partials(lx * lx * val);
    if (AutodiffOrder >= 3) push_partials(lx * lx * lx * val);

    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline var<AutodiffOrder, StrictSmoothness>
    pow(const var<AutodiffOrder, StrictSmoothness>& v1,
        double y) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    create_node<unary_var_node<AutodiffOrder, partials_order>>(n_inputs);
    
    double x = v1.first_val();
    double val = std::pow(x, y);
    
    push_dual_numbers<AutodiffOrder>(val);
    
    push_inputs(v1.dual_numbers());
    
    double lx = std::log(x);
    
    if (AutodiffOrder >= 1) push_partials(val * y / x);
    if (AutodiffOrder >= 2) push_partials( y * (y - 1.0) * val / (x * x) );
    if (AutodiffOrder >= 3) push_partials((y - 2.0) * (y - 1.0) * y * val / (x  * x * x) );
    
    return var<AutodiffOrder, StrictSmoothness>(next_body_idx_ - 1);
    
  }

}

#endif
