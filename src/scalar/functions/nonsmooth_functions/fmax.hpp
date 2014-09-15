#ifndef nomad__src__scalar__functions__nonsmooth_functions__fmax_hpp
#define nomad__src__scalar__functions__nonsmooth_functions__fmax_hpp

#include <math.h>
#include <type_traits>
#include <src/var/var.hpp>

namespace nomad {
  
  double fmax(double x, double y) { return std::fmax(x, y); }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness> >::type
    fmax(const var<AutodiffOrder, StrictSmoothness>& v1,
         const var<AutodiffOrder, StrictSmoothness>& v2) {
    return v1.first_val() > v2.first_val() ? v1 : v2;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness> >::type
    fmax(double x,
         const var<AutodiffOrder, StrictSmoothness>& v2) {
    return x > v2.first_val() ? var<AutodiffOrder, StrictSmoothness>(x) : v2;
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, var<AutodiffOrder, StrictSmoothness> >::type
    fmax(const var<AutodiffOrder, StrictSmoothness>& v1,
         double y) {
    return v1.first_val() > y ? v1 : var<AutodiffOrder, StrictSmoothness>(y);
  }

}

#endif
