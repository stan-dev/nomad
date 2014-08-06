#ifndef nomad__src__scalar__functions__nonsmooth_functions__fmax_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__fmax_hpp

#include <math.h>
#include <type_traits>
#include <src/var/var.hpp>

namespace nomad {
  
  double fmax(double x, double y) { return std::fmax(x, y); }
  
  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, var<autodiff_order, strict_smoothness> >::type
    fmax(const var<autodiff_order, strict_smoothness>& v1,
         const var<autodiff_order, strict_smoothness>& v2) {
    return v1.first_val() > v2.first_val() ? v1 : v2;
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, var<autodiff_order, strict_smoothness> >::type
    fmax(double x,
         const var<autodiff_order, strict_smoothness>& v2) {
    return x > v2.first_val() ? var<autodiff_order, strict_smoothness>(x) : v2;
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, var<autodiff_order, strict_smoothness> >::type
    fmax(const var<autodiff_order, strict_smoothness>& v1,
         double y) {
    return v1.first_val() > y ? v1 : var<autodiff_order, strict_smoothness>(y);
  }

}

#endif
