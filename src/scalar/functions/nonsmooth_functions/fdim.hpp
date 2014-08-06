#ifndef nomad__src__scalar__functions__nonsmooth_functions__fdim_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__fdim_hpp

#include <math.h>
#include <type_traits>

#include <src/var/var.hpp>
#include <src/var/derived/unary_var_body.hpp>
#include <src/var/derived/binary_var_body.hpp>

namespace nomad {
  
  inline double fdim(double x, double y) { return std::fdim(x, y); }
  
  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, var<autodiff_order, strict_smoothness> >::type
    fdim(const var<autodiff_order, strict_smoothness>& v1,
         const var<autodiff_order, strict_smoothness>& v2) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 2;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
    binary_var_body<autodiff_order, partials_order>::n_partials();
    
    new binary_var_body<autodiff_order, partials_order>();
    
    double x = v1.first_val();
    double y = v2.first_val();
    
    push_dual_numbers<autodiff_order>(fdim(x, y));
    
    push_inputs(v1.dual_numbers());
    push_inputs(v2.dual_numbers());
    
    
    if (autodiff_order >= 1) {
      if (x > y) {
        push_partials(1);
        push_partials(-1);
      }
      else {
        push_partials(0);
        push_partials(0);
      }
    }
    
    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, var<autodiff_order, strict_smoothness> >::type
    fdim(double x,
         const var<autodiff_order, strict_smoothness>& v2) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
    unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();
    
    double y = v2.first_val();
    
    push_dual_numbers<autodiff_order>(fdim(x, y));
    
    push_inputs(v2.dual_numbers());
    
    if (autodiff_order >= 1) {
      if (x > y) push_partials(-1);
      else       push_partials(0);
    }
      
    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, var<autodiff_order, strict_smoothness> >::type
    fdim(const var<autodiff_order, strict_smoothness>& v1,
         double y) {
    
    const short partials_order = 1;
    const unsigned int n_inputs = 1;
    
    next_inputs_delta = n_inputs;
    next_partials_delta =
    unary_var_body<autodiff_order, partials_order>::n_partials();
    
    new unary_var_body<autodiff_order, partials_order>();
    
    double x = v1.first_val();
    
    push_dual_numbers<autodiff_order>(fdim(x, y));
    
    push_inputs(v1.dual_numbers());
    
    if (autodiff_order >= 1) {
      if (x > y) push_partials(1);
      else       push_partials(0);
    }

    return var<autodiff_order, strict_smoothness>(next_body_idx_ - 1);
    
  }

}

#endif
