#ifndef nomad__src__scalar__operators__nonsmooth_operators__operator_greater_than_or_equal_to_hpp
#define nomad__src__scalar__operators__nonsmooth_operators__operator_greater_than_or_equal_to_hpp

#include <src/var/var.hpp>

namespace nomad {

  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, bool >::type
    operator>=(const var<autodiff_order, strict_smoothness>& v1,
               const var<autodiff_order, strict_smoothness>& v2) {
    return v1.first_val() >= v2.first_val();
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, bool >::type
    operator>=(double x,
               const var<autodiff_order, strict_smoothness>& v2) {
    return x >= v2.first_val();
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, bool >::type
    operator>=(const var<autodiff_order, strict_smoothness>& v1,
               double y) {
    return v1.first_val() >= y;
  }

}
#endif
