#ifndef nomad__src__scalar__functions__nonsmooth_functions__if_else_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__if_else_hpp

#include <type_traits>
#include <src/var/var.hpp>

namespace nomad {
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness> >::type
    if_else(bool c,
            const var<AutodiffOrder, StrictSmoothness>& v_true,
            const var<AutodiffOrder, StrictSmoothness>& v_false) {
    return c ? v_true : v_false;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness> >::type
  if_else(bool c,
          double x_true,
          const var<AutodiffOrder, StrictSmoothness>& v_false) {
    return c ? x_true : v_false;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness> >::type
  if_else(bool c,
          const var<AutodiffOrder, StrictSmoothness>& v_true,
          double x_false) {
    return c ? v_true : x_false;
  }

}

#endif
