#ifndef nomad__scalar__functions__atan2_hpp
#define nomad__scalar__functions__atan2_hpp

#include <math.h>
#include <var/var.hpp>
#include <var/derived/unary_var_body.hpp>
#include <var/derived/binary_var_body.hpp>

namespace nomad {
  
  inline double atan2(double y, double x) {
    return std::atan2(y, x);
  }
  
  template <short autodiff_order>
  inline var<autodiff_order> atan2(const var<autodiff_order>& v1,
                                   const var<autodiff_order>& v2) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 2;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      binary_var_body<autodiff_order, partials_order>::n_partials();
    
    new binary_var_body<autodiff_order, partials_order>();

    double y = v1.first_val();
    double x = v2.first_val();
    
    push_dual_numbers<autodiff_order>(atan2(y, x));
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    double d = 1.0 / (x * x + y * y);
    
    if (autodiff_order >= 1) {
      push_partials(+ x * d);
      push_partials(- y * d);
    }
    if (autodiff_order >= 2) {
      double d2 = d * d;
      double p = 2 * x * y * d2;
      
      push_partials(-p);
      push_partials((y * y - x * x) * d2);
      push_partials(p);
    }
    if (autodiff_order >= 3) {
      double d3 = d * d * d;
      double p1 = 2 * y * (y * y - 3 * x * x) * d3;
      double p2 = 2 * x * (x * x - 3 * y * y) * d3;
      
      push_partials(- p2);
      push_partials(- p1);
      push_partials(p2);
      push_partials(p1);
    }

    return var<autodiff_order>(next_body_idx_ - 1);
    
  }
  
  template <short autodiff_order>
  inline var<autodiff_order> atan2(double y,
                                   const var<autodiff_order>& v2) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();
    
    double x = v2.first_val();
    
    push_dual_numbers<autodiff_order>(atan2(y, x));
    
    push_inputs(v2.dual_numbers());
    
    double d = 1.0 / (x * x + y * y);
    
    if (autodiff_order >= 1) push_partials(- y * d);
    if (autodiff_order >= 2) push_partials(2 * x * y * d * d);
    if (autodiff_order >= 3) push_partials(2 * y * (y * y - 3 * x * x) * d * d * d);
    
    return var<autodiff_order>(next_body_idx_ - 1);
    
  }
  
  template <short autodiff_order>
  inline var<autodiff_order> atan2(const var<autodiff_order>& v1,
                                   double x) {
    
    const short partials_order = 3;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
      unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();
    
    double y = v1.first_val();
    
    push_dual_numbers<autodiff_order>(atan2(y, x));
    
    push_inputs(v1.dual_numbers());
    
    double d = 1.0 / (x * x + y * y);
    
    if (autodiff_order >= 1) push_partials(x * d);
    if (autodiff_order >= 2) push_partials(- 2 * x * y * d * d);
    if (autodiff_order >= 3) push_partials(- 2 * x * (x * x - 3 * y * y) * d * d * d);
    
    return var<autodiff_order>(next_body_idx_ - 1);
    
  }

}

#endif
