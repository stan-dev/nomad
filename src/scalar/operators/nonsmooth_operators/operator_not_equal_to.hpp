#ifndef nomad__src__scalar__operators__nonsmooth_operators__operator_not_equal_to_hpp
#define nomad__src__scalar__operators__nonsmooth_operators__operator_not_equal_to_hpp

#include <src/var/var.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, bool >::type
    operator!=(const var<AutodiffOrder, StrictSmoothness>& v1,
               const var<AutodiffOrder, StrictSmoothness>& v2) {
    return v1.first_val() != v2.first_val();
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, bool >::type
    operator!=(double x,
               const var<AutodiffOrder, StrictSmoothness>& v2) {
    return x != v2.first_val();
  }
  
  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, bool >::type
    operator!=(const var<AutodiffOrder, StrictSmoothness>& v1,
               double y) {
    return v1.first_val() != y;
  }

}
#endif
