#ifndef nomad__src__scalar__operators__nonsmooth_operators__operator_unary_not_hpp
#define nomad__src__scalar__operators__nonsmooth_operators__operator_unary_not_hpp

#include <src/var/var.hpp>

namespace nomad {

  template <short AutodiffOrder, bool StrictSmoothness>
  inline typename std::enable_if<!StrictSmoothness, bool >::type
    operator!(const var<AutodiffOrder, StrictSmoothness>& input) {
    return !(input.first_val());
  }

}
#endif
