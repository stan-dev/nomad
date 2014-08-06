#ifndef nomad__src__scalar__functions__nonsmooth_functions__if_else_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__if_else_hpp

#include <type_traits>
#include <src/var/var.hpp>

namespace nomad {
  
  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, var<autodiff_order, strict_smoothness> >::type
    if_else(bool c,
            const var<autodiff_order, strict_smoothness>& v_true,
            const var<autodiff_order, strict_smoothness>& v_false) {
    return c ? v_true : v_false;
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, var<autodiff_order, strict_smoothness> >::type
  if_else(bool c,
          double x_true,
          const var<autodiff_order, strict_smoothness>& v_false) {
    return c ? x_true : v_false;
  }
  
  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, var<autodiff_order, strict_smoothness> >::type
  if_else(bool c,
          const var<autodiff_order, strict_smoothness>& v_true,
          double x_false) {
    return c ? v_true : x_false;
  }

}

#endif
