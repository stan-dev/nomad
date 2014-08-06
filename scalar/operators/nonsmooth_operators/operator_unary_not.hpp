#ifndef nomad__scalar__operators__nonsmooth_operators__operator_unary_not_hpp
#define nomad__scalar__operators__nonsmooth_operators__operator_unary_not_hpp

#include <var/var.hpp>

namespace nomad {

  template <short autodiff_order, bool strict_smoothness>
  inline typename std::enable_if<!strict_smoothness, bool >::type
    operator!(const var<autodiff_order, strict_smoothness>& input) {
    return !(input.first_val());
  }

}
#endif
